import { Buffer } from "buffer";
import { Connection, Keypair, PublicKey, TransactionMessage, VersionedTransaction } from "@solana/web3.js";
import type { TransactionInstruction } from "@solana/web3.js";
import type { DevnetConfig } from "./config";
import { effectiveAbsQ, parseMarket, parsePortfolio } from "./accounts";
import { layout } from "./layout";
import { computeBudgetIxs, crankIx, createPortfolioIxs, depositIx, tradeCpiIx, withdrawIx } from "./instructions";
import { ata, matcherDelegate, vault, vaultAuthority } from "./pda";

const MAX_CRANKS = 8;

export async function findPortfolio(conn: Connection, programId: PublicKey, market: { marketGroupId: Uint8Array }, owner: PublicKey): Promise<PublicKey | null> {
  const accounts = await conn.getProgramAccounts(programId, { filters: [
    { dataSize: layout.PORTFOLIO_ACCOUNT_LEN },
    { memcmp: { offset: layout.HEADER_LEN + layout["pf.provenance_header"], bytes: new PublicKey(market.marketGroupId).toBase58() } },
    { memcmp: { offset: layout.HEADER_LEN + layout["pf.owner"], bytes: owner.toBase58() } },
  ] });
  return accounts[0]?.pubkey ?? null;
}

async function marketPortfolios(conn: Connection, programId: PublicKey, market: { marketGroupId: Uint8Array }) {
  return conn.getProgramAccounts(programId, { filters: [
    { dataSize: layout.PORTFOLIO_ACCOUNT_LEN },
    { memcmp: { offset: layout.HEADER_LEN + layout["pf.provenance_header"], bytes: new PublicKey(market.marketGroupId).toBase58() } },
  ] });
}

const isPositioned = (data: Uint8Array) => parsePortfolio(data).legs.some(leg => leg.assetIndex === 0 && leg.basisPosQ !== 0n);

export async function positionedPortfolios(conn: Connection, programId: PublicKey, market: { marketGroupId: Uint8Array }): Promise<PublicKey[]> {
  return (await marketPortfolios(conn, programId, market)).filter(({ account }) => isPositioned(account.data)).map(({ pubkey }) => pubkey);
}

/// Simulates one crank per portfolio after the cranks already kept and keeps those that pass.
/// Given `before` (portfolio data by base58 key), a crank is kept only if it also changes its
/// portfolio: one that passes only by advancing the market clock is a no-op once a keeper cranks
/// in the same slot, and the wrapper then fails the whole transaction with EngineNonProgress.
export async function withFreshCrank(
  conn: Connection, payer: PublicKey, programId: PublicKey, market: PublicKey,
  portfolios: PublicKey[], readAccounts: PublicKey[], before?: Map<string, Uint8Array>,
): Promise<{ ixs: TransactionInstruction[]; accounts: Uint8Array[] }> {
  const kept: TransactionInstruction[] = [];
  const budget = computeBudgetIxs();
  const nowSlot = await conn.getSlot();
  const { blockhash } = await conn.getLatestBlockhash();
  for (const portfolio of portfolios) {
    const crank = crankIx(programId, payer, market, portfolio, {
      nowSlot, observations: [{ assetIndex: 0, oracleAccounts: 0 }],
    });
    const tx = new VersionedTransaction(new TransactionMessage({
      payerKey: payer, recentBlockhash: blockhash, instructions: [...budget, ...kept, crank],
    }).compileToV0Message());
    const result = await conn.simulateTransaction(tx, { sigVerify: false, replaceRecentBlockhash: true,
      ...(before ? { accounts: { encoding: "base64" as const, addresses: [portfolio.toBase58()] } } : {}) });
    if (result.value.err) continue;
    const prior = before?.get(portfolio.toBase58()), after = result.value.accounts?.[0];
    if (prior && after && Buffer.from(after.data[0], "base64").equals(Buffer.from(prior))) continue;
    kept.push(crank);
  }
  const ixs = [...budget, ...kept];
  if (!kept.length) {
    const accounts = await conn.getMultipleAccountsInfo(readAccounts);
    return { ixs, accounts: accounts.map(account => {
      if (!account) throw new Error("Market refresh: account not found");
      return account.data;
    }) };
  }
  const tx = new VersionedTransaction(new TransactionMessage({
    payerKey: payer, recentBlockhash: blockhash, instructions: ixs,
  }).compileToV0Message());
  const result = await conn.simulateTransaction(tx, {
    sigVerify: false, replaceRecentBlockhash: true,
    accounts: { encoding: "base64", addresses: readAccounts.map(key => key.toBase58()) },
  });
  if (result.value.err) throw new Error(`Market refresh simulation failed: ${JSON.stringify(result.value.err)}`);
  const accounts = result.value.accounts;
  if (!accounts || accounts.length !== readAccounts.length) throw new Error("Market refresh: accounts missing");
  return { ixs, accounts: accounts.map(account => {
    if (!account) throw new Error("Market refresh: account not found");
    return Buffer.from(account.data[0], "base64");
  }) };
}

async function readPortfolio(conn: Connection, portfolio: PublicKey) {
  const account = await conn.getAccountInfo(portfolio);
  if (!account) throw new Error("Portfolio account not found");
  return parsePortfolio(account.data);
}

async function crankSet(conn: Connection, cfg: DevnetConfig, first: PublicKey[], exclude: PublicKey[] = []) {
  const marketAccount = await conn.getAccountInfo(cfg.market);
  if (!marketAccount) throw new Error("Market account not found");
  const market = parseMarket(marketAccount.data);
  const all = await marketPortfolios(conn, cfg.programId, market);
  const skip = new Set(exclude.map(key => key.toBase58()));
  const positioned = all.filter(({ account }) => isPositioned(account.data)).map(({ pubkey }) => pubkey);
  const portfolios = [...new Set([...first, ...positioned].map(key => key.toBase58()))]
    .filter(key => !skip.has(key)).map(key => new PublicKey(key));
  if (portfolios.length > MAX_CRANKS) throw new Error(`${portfolios.length} portfolios need a market refresh; more than ${MAX_CRANKS} fit in one transaction`);
  return { portfolios, market, before: new Map(all.map(({ pubkey, account }) => [pubkey.toBase58(), account.data])) };
}

export async function buildDeposit(conn: Connection, cfg: DevnetConfig, owner: PublicKey, portfolio: PublicKey, amountAtoms: bigint) {
  const pf = await readPortfolio(conn, portfolio);
  return [...computeBudgetIxs(), depositIx(cfg.programId, owner, cfg.market, portfolio,
    ata(owner, cfg.mint), vault(cfg.programId, cfg.market, cfg.mint),
    { portfolioId: pf.portfolioId, expectedSequence: pf.sequence, amount: amountAtoms })];
}

export async function buildWithdraw(conn: Connection, cfg: DevnetConfig, owner: PublicKey, portfolio: PublicKey, amountAtoms: bigint) {
  const { portfolios } = await crankSet(conn, cfg, [portfolio]);
  const { ixs, accounts } = await withFreshCrank(conn, owner, cfg.programId, cfg.market, portfolios, [portfolio]);
  const pf = parsePortfolio(accounts[0]);
  return [...ixs, withdrawIx(cfg.programId, owner, cfg.market, portfolio,
    ata(owner, cfg.mint), vault(cfg.programId, cfg.market, cfg.mint), vaultAuthority(cfg.programId, cfg.market),
    { portfolioId: pf.portfolioId, expectedSequence: pf.sequence, amount: amountAtoms })];
}

export async function buildTrade(
  conn: Connection, cfg: DevnetConfig, owner: PublicKey, portfolio: PublicKey,
  sizeQ: bigint, limitPrice: bigint, feeBps: bigint,
) {
  return buildTradeIxs(conn, cfg, owner, portfolio, () => ({ sizeQ, limitPrice }), feeBps);
}

export async function buildClose(
  conn: Connection, cfg: DevnetConfig, owner: PublicKey, portfolio: PublicKey,
  slippageBps: bigint, feeBps: bigint, maxSizeQ?: bigint,
) {
  return buildTradeIxs(conn, cfg, owner, portfolio, (pf, market) => {
    const leg = pf.legs.find(leg => leg.assetIndex === 0 && leg.marketId === market.asset0.marketId);
    let q = leg ? effectiveAbsQ(market, leg) : 0n;
    if (!leg || q === 0n) throw new Error("no open position");
    if (maxSizeQ !== undefined) q = q > maxSizeQ ? maxSizeQ : q;
    const price = market.asset0.effectivePrice;
    const slippage = price * slippageBps / 10000n;
    return { sizeQ: leg.side === "long" ? -q : q, limitPrice: leg.side === "long" ? price - slippage : price + slippage };
  }, feeBps);
}

async function buildTradeIxs(
  conn: Connection, cfg: DevnetConfig, owner: PublicKey, portfolio: PublicKey,
  order: (pf: ReturnType<typeof parsePortfolio>, market: ReturnType<typeof parseMarket>) => { sizeQ: bigint; limitPrice: bigint },
  feeBps: bigint,
) {
  if (!cfg.matcherProgram || !cfg.matcherContext) throw new Error("Trading is waiting on the matcher program (devnet PoC)");
  // TradeCpi settles its own two portfolios before the fill (engine v16.rs
  // execute_batch_with_fee_after_tail_validation_not_atomic), so the LP is never cranked: it trades
  // with every agent, so its crank was the likeliest to land as a no-op. The other positioned
  // portfolios are cranked when stale, since a risk-increasing fill fails EngineLockActive while
  // any is stale after a mark or funding move (trade_preflight_risk_gate). The taker's crank is
  // simulated last so the order is sized from its settled state, but sent only when no other crank
  // runs and it walks the mark to a pushed oracle target, which the same gate also requires.
  const { portfolios: others, market: prior, before } = await crankSet(conn, cfg, [], [portfolio, cfg.lpPortfolio]);
  before.delete(portfolio.toBase58()); // a flat taker's crank changes no portfolio but may still walk the mark
  const fresh = await withFreshCrank(conn, owner, cfg.programId, cfg.market,
    [...others, portfolio], [portfolio, cfg.lpPortfolio, cfg.market], before);
  const { accounts } = fresh;
  let { ixs } = fresh;
  const cranks = ixs.length - computeBudgetIxs().length;
  if (cranks && ixs.at(-1)!.keys[2].pubkey.equals(portfolio)
    && (cranks > 1 || parseMarket(accounts[2]).asset0.effectivePrice === prior.asset0.effectivePrice)) ixs = ixs.slice(0, -1);
  const a = parsePortfolio(accounts[0]), b = parsePortfolio(accounts[1]);
  const market = parseMarket(accounts[2]);
  const { sizeQ, limitPrice } = order(a, market);
  // The wrapper refuses a TradeCpi fee under the market's trade_fee_base_bps, and markets
  // differ (12 bps on the first SOL-PERP market, 15 on COMPUTE-PERP), so never send less than the market's own.
  const fee = feeBps < market.config.tradeFeeBaseBps ? market.config.tradeFeeBaseBps : feeBps;
  const delegate = matcherDelegate(cfg.programId, cfg.market, cfg.lpPortfolio, b.owner, cfg.matcherProgram, cfg.matcherContext);
  return [...ixs, tradeCpiIx(cfg.programId, owner, cfg.market, portfolio, cfg.lpPortfolio,
    cfg.matcherProgram, cfg.matcherContext, delegate, {
      accountAPortfolioId: a.portfolioId, accountAPositionEpoch: a.positionEpoch,
      accountBPortfolioId: b.portfolioId, accountBPositionEpoch: b.positionEpoch, accountBMatcherSequence: b.sequence,
      assetIndex: 0, marketId: market.asset0.marketId, sizeQ, feeBps: fee, limitPrice, backingFeeCapBps: 0,
    })];
}

export async function buildCreatePortfolio(cfg: DevnetConfig, owner: PublicKey) {
  const portfolio = Keypair.generate();
  const ixs = await createPortfolioIxs(cfg.programId, owner, cfg.market, portfolio.publicKey);
  return { ixs: [...computeBudgetIxs(), ...ixs], portfolio };
}
