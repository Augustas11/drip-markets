// @vitest-environment node
import { readFileSync } from "node:fs";
import { Connection, PublicKey } from "@solana/web3.js";
import { afterEach, expect, it, vi } from "vitest";
import { devnetConfig } from "./config";
import { buildClose, buildDeposit, buildTrade, buildWithdraw, findPortfolio, positionedPortfolios, withFreshCrank } from "./tx";
import { effectiveAbsQ, parseMarket, parsePortfolio } from "./accounts";
import { i128, u64 } from "./bytes";
import { layout } from "./layout";
import { matcherDelegate } from "./pda";
import pdas from "./fixtures/pdas.json";

const conn = new Connection(devnetConfig.rpcUrl);
const cfg = { ...devnetConfig, matcherProgram: new PublicKey(pdas.matcherDelegate.matcherProgram), matcherContext: new PublicKey(pdas.matcherDelegate.matcherContext) };
const portfolio = new PublicKey(new Uint8Array(32).fill(4));
const data = readFileSync(new URL("./fixtures/portfolio.bin", import.meta.url));
const lp = readFileSync(new URL("./fixtures/lp_portfolio.bin", import.meta.url));
const market = readFileSync(new URL("./fixtures/market.bin", import.meta.url));
const owner = parsePortfolio(data).owner;
const account = (data: Buffer) => ({ data, owner: cfg.programId, executable: false, lamports: 1, rentEpoch: 0 });

function simulatedAccounts(buffers: Buffer[]) {
  return buffers.map(data => ({ ...account(data), owner: cfg.programId.toBase58(), data: [data.toString("base64"), "base64"] }));
}

function mockCranks(buffers: Buffer[] = [data, lp, market]) {
  vi.spyOn(conn, "getAccountInfo").mockResolvedValue(account(market));
  vi.spyOn(conn, "getMultipleAccountsInfo").mockImplementation(async keys => keys.map(key =>
    account(key.equals(cfg.market) ? market : key.equals(cfg.lpPortfolio) ? lp : data)));
  vi.spyOn(conn, "getProgramAccounts").mockResolvedValue([]);
  vi.spyOn(conn, "getSlot").mockResolvedValue(123);
  vi.spyOn(conn, "getLatestBlockhash").mockResolvedValue({ blockhash: cfg.market.toBase58(), lastValidBlockHeight: 1234 });
  return vi.spyOn(conn, "simulateTransaction").mockResolvedValue({ context: { slot: 123 }, value: { err: null, logs: [], accounts: simulatedAccounts(buffers) } });
}
afterEach(() => vi.restoreAllMocks());

it("finds portfolios using the owner offset and account length", async () => {
  const get = vi.spyOn(conn, "getProgramAccounts").mockResolvedValue([{ pubkey: portfolio, account: account(data) }]);
  expect(await findPortfolio(conn, cfg.programId, parseMarket(market), owner)).toEqual(portfolio);
  expect(get).toHaveBeenCalledWith(cfg.programId, { filters: [
    { dataSize: layout.PORTFOLIO_ACCOUNT_LEN },
    { memcmp: { offset: layout.HEADER_LEN + layout["pf.provenance_header"], bytes: new PublicKey(parseMarket(market).marketGroupId).toBase58() } },
    { memcmp: { offset: layout.HEADER_LEN + layout["pf.owner"], bytes: owner.toBase58() } },
  ] });
  get.mockResolvedValue([]);
  expect(await findPortfolio(conn, cfg.programId, parseMarket(market), owner)).toBeNull();
});

it("simulates each crank after previously kept cranks and drops a no-op", async () => {
  const simulate = mockCranks();
  simulate.mockResolvedValueOnce({ context: { slot: 123 }, value: { err: { InstructionError: [2, "InvalidArgument"] }, logs: [] } });
  const { ixs, accounts } = await withFreshCrank(conn, owner, cfg.programId, cfg.market, [portfolio, cfg.lpPortfolio, portfolio], [portfolio, cfg.lpPortfolio, cfg.market]);
  expect(accounts).toEqual([data, lp, market]);
  expect(ixs.map(ix => ix.data[0])).toEqual([2, 1, 5, 5]);
  expect(simulate.mock.calls.map(call => call[0].message.compiledInstructions.length)).toEqual([3, 3, 4, 4]);
  expect(simulate.mock.calls[0][1]).toEqual({ sigVerify: false, replaceRecentBlockhash: true });
});

it("finds asset-0 positions without an owner filter and drops flat portfolios", async () => {
  const flat = Buffer.from(data), otherAsset = Buffer.from(data), inactive = Buffer.from(data);
  const leg = layout.HEADER_LEN + layout["pf.legs"];
  flat.fill(0, leg + layout["leg.basis_pos_q"], leg + layout["leg.basis_pos_q"] + 16);
  otherAsset.writeUInt32LE(1, leg + layout["leg.asset_index"]);
  inactive[leg + layout["leg.active"]] = 0;
  const get = vi.spyOn(conn, "getProgramAccounts").mockResolvedValue([
    { pubkey: cfg.lpPortfolio, account: account(flat) },
    { pubkey: portfolio, account: account(data) },
    { pubkey: cfg.lpPortfolio, account: account(otherAsset) },
    { pubkey: cfg.lpPortfolio, account: account(inactive) },
  ]);
  expect(await positionedPortfolios(conn, cfg.programId, parseMarket(market))).toEqual([portfolio]);
  expect(get).toHaveBeenCalledWith(cfg.programId, { filters: [
    { dataSize: layout.PORTFOLIO_ACCOUNT_LEN },
    { memcmp: { offset: layout.HEADER_LEN + layout["pf.provenance_header"], bytes: new PublicKey(parseMarket(market).marketGroupId).toBase58() } },
  ] });
});

const priced = (price: bigint) => {
  const out = Buffer.from(market);
  out.writeBigUInt64LE(price, layout.ASSET0_SLOT_OFF + layout.ASSET_ORACLE_WRAPPER_LEN + layout["slot.asset"] + layout["asset.effective_price"]);
  return out;
};
const crankTargets = (ixs: { programId: PublicKey; data: Buffer; keys: { pubkey: PublicKey }[] }[]) =>
  ixs.filter(ix => ix.programId.equals(cfg.programId) && ix.data[0] === 5).map(ix => ix.keys[2].pubkey);
const settled = Buffer.from(data);
settled.writeBigUInt64LE(77n, layout.HEADER_LEN + layout["pf.capital"]);
const simulated = (buffers: Buffer[]) => ({ context: { slot: 123 }, value: { err: null, logs: [], accounts: simulatedAccounts(buffers) } });

it("cranks other positioned portfolios only when the crank changes them, never the LP", async () => {
  const simulate = mockCranks();
  const third = new PublicKey(new Uint8Array(32).fill(5)), fourth = new PublicKey(new Uint8Array(32).fill(6));
  vi.mocked(conn.getProgramAccounts).mockResolvedValue([portfolio, cfg.lpPortfolio, third, fourth, third].map(pubkey => ({ pubkey, account: account(data) })));
  // third is settled by its crank; fourth passes only by advancing the clock; the taker is settled too.
  simulate.mockResolvedValueOnce(simulated([settled])).mockResolvedValueOnce(simulated([data])).mockResolvedValueOnce(simulated([settled]));
  const ixs = await buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 12n);
  expect(crankTargets(ixs)).toEqual([third]);
  expect(simulate.mock.calls.slice(0, 3).map(call => call[1])).toEqual([third, fourth, portfolio].map(key => ({
    sigVerify: false, replaceRecentBlockhash: true, accounts: { encoding: "base64", addresses: [key.toBase58()] },
  })));
  expect(simulate.mock.calls.map(call => call[0].message.compiledInstructions.length)).toEqual([3, 4, 4, 4]);
  expect(conn.getMultipleAccountsInfo).not.toHaveBeenCalled();
});

it("sends the taker's crank only when it alone walks the mark", async () => {
  const simulate = mockCranks();
  vi.mocked(conn.getProgramAccounts).mockResolvedValue([portfolio, cfg.lpPortfolio].map(pubkey => ({ pubkey, account: account(data) })));
  simulate.mockResolvedValueOnce(simulated([settled]));
  expect(crankTargets(await buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 12n))).toEqual([]);
  mockCranks([settled, lp, priced(150_000_000n)]);
  vi.mocked(conn.getProgramAccounts).mockResolvedValue([portfolio, cfg.lpPortfolio].map(pubkey => ({ pubkey, account: account(data) })));
  expect(crankTargets(await buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 12n))).toEqual([portfolio]);
});

it("drops the taker's crank behind another crank even when the mark moves", async () => {
  mockCranks([settled, lp, priced(150_000_000n)]);
  const third = new PublicKey(new Uint8Array(32).fill(5));
  vi.mocked(conn.getProgramAccounts).mockResolvedValue([portfolio, third].map(pubkey => ({ pubkey, account: account(data) })));
  expect(crankTargets(await buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 12n))).toEqual([third]);
});

it("keeps an unchanged taker's crank when it walks the mark", async () => {
  mockCranks([data, lp, priced(150_000_000n)]);
  vi.mocked(conn.getProgramAccounts).mockResolvedValue([portfolio, cfg.lpPortfolio].map(pubkey => ({ pubkey, account: account(data) })));
  expect(crankTargets(await buildTrade(conn, cfg, owner, portfolio, 123n, 100000000n, 12n))).toEqual([portfolio]);
});

it("reads the chain when no crank passes", async () => {
  const simulate = mockCranks();
  const third = new PublicKey(new Uint8Array(32).fill(5));
  vi.mocked(conn.getProgramAccounts).mockResolvedValue([portfolio, cfg.lpPortfolio, third].map(pubkey => ({ pubkey, account: account(data) })));
  simulate.mockResolvedValue({ context: { slot: 123 }, value: { err: { InstructionError: [2, { Custom: 22 }] }, logs: [] } });
  const ixs = await buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 12n);
  expect(crankTargets(ixs)).toEqual([]);
  expect(simulate).toHaveBeenCalledTimes(2);
  expect(conn.getMultipleAccountsInfo).toHaveBeenCalledExactlyOnceWith([portfolio, cfg.lpPortfolio, cfg.market]);
});

it("rejects a trade needing nine cranks before simulation", async () => {
  const simulate = mockCranks();
  const portfolios = [portfolio, cfg.lpPortfolio, ...Array.from({ length: 9 }, (_, i) => new PublicKey(new Uint8Array(32).fill(i + 5)))];
  vi.mocked(conn.getProgramAccounts).mockResolvedValue(portfolios.map(pubkey => ({ pubkey, account: account(data) })));
  await expect(buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 12n))
    .rejects.toThrow("9 portfolios need a market refresh; more than 8 fit in one transaction");
  expect(simulate).not.toHaveBeenCalled();
});

it("cranks other positioned portfolios once when building a withdraw", async () => {
  mockCranks([data]);
  const third = new PublicKey(new Uint8Array(32).fill(5));
  vi.mocked(conn.getProgramAccounts).mockResolvedValue([portfolio, third, third].map(pubkey => ({ pubkey, account: account(data) })));
  const ixs = await buildWithdraw(conn, cfg, owner, portfolio, 789n);
  expect(ixs.filter(ix => ix.programId.equals(cfg.programId) && ix.data[0] === 5).map(ix => ix.keys[2].pubkey))
    .toEqual([portfolio, third]);
});

it("reads a new sequence on every deposit and withdraw build", async () => {
  const changed = Buffer.from(data);
  changed.writeBigUInt64LE(987n, layout.PORTFOLIO_MATCHER_SEQUENCE_OFF);
  mockCranks([changed]);
  const read = vi.mocked(conn.getAccountInfo).mockResolvedValueOnce(account(data)).mockResolvedValueOnce(account(changed));
  const first = await buildDeposit(conn, cfg, owner, portfolio, 123n);
  const second = await buildDeposit(conn, cfg, owner, portfolio, 456n);
  const withdraw = await buildWithdraw(conn, cfg, owner, portfolio, 789n);
  expect(u64(first.at(-1)!.data, 9)).toBe(1n);
  expect(u64(second.at(-1)!.data, 9)).toBe(987n);
  expect(u64(withdraw.at(-1)!.data, 9)).toBe(987n);
  expect(read).toHaveBeenCalledTimes(3);
  expect(read).toHaveBeenLastCalledWith(cfg.market);
  expect(withdraw.at(-2)!.data[0]).toBe(5);
});

it("builds a fresh TradeCpi binding with the LP's runtime owner and no LP signer", async () => {
  const advanced = Buffer.from(data);
  advanced.writeBigUInt64LE(32n, layout.PORTFOLIO_MATCHER_CONTROL_OFF);
  const simulate = mockCranks([advanced, lp, market]);
  const read = vi.mocked(conn.getMultipleAccountsInfo);
  const ixs = await buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 12n);
  const trade = ixs.at(-1)!;
  expect(trade.data[0]).toBe(10);
  expect(u64(trade.data, 1)).toBe(2n);
  expect(u64(trade.data, 9)).toBe(16n);
  expect(u64(trade.data, 17)).toBe(1n);
  expect(u64(trade.data, 25)).toBe(15n);
  expect(u64(trade.data, 33)).toBe(1n);
  expect(i128(trade.data, 51)).toBe(-123n);
  expect(trade.keys.filter(key => key.isSigner).map(key => key.pubkey)).toEqual([owner]);
  expect(trade.keys[6].pubkey).toEqual(matcherDelegate(cfg.programId, cfg.market, cfg.lpPortfolio,
    parsePortfolio(lp).owner, cfg.matcherProgram, cfg.matcherContext));
  expect(read).not.toHaveBeenCalled();
  expect(simulate.mock.calls.at(-1)![1]).toEqual({
    sigVerify: false, replaceRecentBlockhash: true,
    accounts: { encoding: "base64", addresses: [portfolio, cfg.lpPortfolio, cfg.market].map(key => key.toBase58()) },
  });
  await expect(buildTrade(conn, { ...cfg, matcherContext: undefined }, owner, portfolio, 1n, 1n, 12n))
    .rejects.toThrow("Trading is waiting on the matcher program (devnet PoC)");
});

it("reads accounts once when no crank is kept", async () => {
  mockCranks().mockResolvedValue({ context: { slot: 123 }, value: { err: "InvalidArgument", logs: [] } });
  const read = vi.mocked(conn.getMultipleAccountsInfo).mockResolvedValue([account(data)]);
  const { ixs, accounts } = await withFreshCrank(conn, owner, cfg.programId, cfg.market, [portfolio], [portfolio]);
  expect(ixs.map(ix => ix.data[0])).toEqual([2, 1]);
  expect(accounts).toEqual([data]);
  expect(read).toHaveBeenCalledExactlyOnceWith([portfolio]);
});

it.each(["long", "short"] as const)("closes the post-crank %s size and binds its epoch and mark", async side => {
  const advanced = Buffer.from(data), changedMarket = Buffer.from(market);
  const leg = layout.HEADER_LEN + layout["pf.legs"];
  advanced[leg + layout["leg.side"]] = side === "long" ? 0 : 1;
  advanced.writeBigInt64LE(side === "long" ? 4_000_000n : -4_000_000n, leg + layout["leg.basis_pos_q"]);
  advanced.writeBigInt64LE(side === "long" ? 0n : -1n, leg + layout["leg.basis_pos_q"] + 8);
  advanced.writeBigUInt64LE(32n, layout.PORTFOLIO_MATCHER_CONTROL_OFF);
  const asset = layout.ASSET0_SLOT_OFF + layout.ASSET_ORACLE_WRAPPER_LEN + layout["slot.asset"];
  changedMarket.writeBigUInt64LE(120_000_000n, asset + layout["asset.effective_price"]);
  mockCranks([advanced, lp, changedMarket]);
  const read = vi.mocked(conn.getMultipleAccountsInfo);
  const trade = (await buildClose(conn, cfg, owner, portfolio, 50n, 12n)).at(-1)!;
  const freshMarket = parseMarket(changedMarket);
  const q = effectiveAbsQ(freshMarket, parsePortfolio(advanced).legs[0]);
  expect(q).toBeGreaterThan(0n);
  expect(q).toBeLessThan(effectiveAbsQ(parseMarket(market), parsePortfolio(data).legs[0]));
  expect(i128(trade.data, 51)).toBe(side === "long" ? -q : q);
  for (const maxSizeQ of [q / 2n, q, q * 2n]) {
    const capped = (await buildClose(conn, cfg, owner, portfolio, 50n, 12n, maxSizeQ)).at(-1)!;
    const expected = q > maxSizeQ ? maxSizeQ : q;
    expect(i128(capped.data, 51)).toBe(side === "long" ? -expected : expected);
    expect(u64(capped.data, 75)).toBe(u64(trade.data, 75));
  }
  expect(u64(trade.data, 9)).toBe(parsePortfolio(advanced).positionEpoch);
  expect(u64(trade.data, 43)).toBe(freshMarket.asset0.marketId);
  expect(u64(trade.data, 67)).toBe(12n);
  expect(u64(trade.data, 75)).toBe(side === "long" ? 119_400_000n : 120_600_000n);
  expect(read).not.toHaveBeenCalled();
});

it("never sends a fee under the market's base fee", async () => {
  const dearer = Buffer.from(market);
  dearer.writeBigUInt64LE(15n, layout.HEADER_LEN + layout["cfg.trade_fee_base_bps"]);
  expect(parseMarket(dearer).config.tradeFeeBaseBps).toBe(15n);
  mockCranks([data, lp, dearer]);
  const trade = (await buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 12n)).at(-1)!;
  expect(u64(trade.data, 67)).toBe(15n);
  mockCranks([data, lp, market]);
  const kept = (await buildTrade(conn, cfg, owner, portfolio, -123n, 100000000n, 20n)).at(-1)!;
  expect(u64(kept.data, 67)).toBe(20n);
});
