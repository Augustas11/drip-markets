/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// VITE_DEV_LIVE=1: the dev server forwards the feed, indexer and faucet paths to the live devnet site.
const LIVE = "https://drip-devnet.vercel.app";

export default defineConfig(({ mode }) => {
  const live = loadEnv(mode, process.cwd(), "").VITE_DEV_LIVE === "1";
  return {
    plugins: [react()],
    server: live ? { proxy: Object.fromEntries(["/compute-feed", "/indexer", "/faucet"].map(path => [path, { target: LIVE, changeOrigin: true }])) } : undefined,
    test: { environment: "jsdom" },
  };
});
