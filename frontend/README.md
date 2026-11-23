# SOL-PERP PoC frontend

From `frontend/`, run `npm ci`, then `npm run dev` for the deterministic mock,
or `VITE_MARKET_CLIENT=devnet npm run dev` for devnet with Phantom or Solflare.
Copy `env.example` to `.env.local` (`env.production.example` is the live
site's build env: copy it to `.env.production`) to override the RPC and account addresses.

Fund your wallet with devnet SOL using `solana airdrop 2 <wallet> --url devnet`
or https://faucet.solana.com. For test USDC, use the faucet on the site's trade page (`faucet/` is its server).
Connect and deposit to create a portfolio.

Trades are `TradeCpi` against our LP's devnet matcher program, which is the
default in `src/chain/config.ts`. The tester wallet is the only signer. Fills
need the mark pusher running; when it is down, the matcher refuses with custom
error 9 (stale) and the engine with `EngineLockActive`.
Entry price and unrealized PnL display as “—” because the engine stores no entry
price. Fills show only this client's landed transactions and the mark at send time.

`npm test` runs tests; `npm run build` typechecks and builds for production.

In devnet mode, `VITE_INDEXER_URL` defaults to `http://127.0.0.1:8090`; set it to `none` to disable the read-only indexer panels.
From the repository root, run `cargo run --release --manifest-path indexer/Cargo.toml -- run`.
The indexer listens on `127.0.0.1:8090`; CORS allows `localhost:5173` by default.
Panels poll every 15 seconds. All figures are devnet, test tokens; mock mode uses deterministic indexer data.

`src/chain/` is the client for upstream Percolator v16 (the wrapper in
`scripts/pins.env`). Its tests match byte for byte the fixtures that the Rust
`client/` crate emits from the pinned program crate. Regenerate them from the
repository root on every re-pin with
`cargo run --manifest-path client/Cargo.toml --bin ts_fixtures -- frontend/src/chain/fixtures`.
The `.bin` account dumps must have the pinned layout, so refresh any whose
layout changed first with `solana account <key> --output-file frontend/src/chain/fixtures/<name>.bin`
(`market.bin` = market, `portfolio.bin` = a trader portfolio, `lp_portfolio.bin` = the LP portfolio).
All three dumps come from one local-validator run, so they share one market
group: the `staging step` sequence (`init-market` through `handover`) on
`launch/sol-perp.toml` with `initial_price = 99938164` and
`trade_fee_base_bps = 12` (the earlier devnet market's values), then
`staging bots` until the matcher's first trade gave the trader and the LP
their legs.

`npm run roundtrip` runs a real devnet round trip. It creates a portfolio,
deposits, opens, closes and withdraws, then closes the portfolio to refund rent
(unless `ROUNDTRIP_KEEP_PORTFOLIO=1`). It prints signatures and saves progress
after each send to `roundtrip-result.json` in `frontend/`. Failed builds or simulations
retry up to 9 times, 3 seconds apart, before aborting without sending. It needs a
funded throwaway wallet (`TESTER_KEYPAIR`). Its trades take the same `TradeCpi`
path as the browser.
Test tokens only, no real funds.
