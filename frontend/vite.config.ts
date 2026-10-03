// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import { thirdPartyLicenses } from "./scripts/third-party-licenses.js";

// Every API prefix the dashboard calls. The dev server forwards them to a
// running giq; the built page is served by giq itself and needs none of this.
const API_PREFIXES = [
  "/status",
  "/gpus",
  "/stats",
  "/storage",
  "/control",
  "/engines",
  "/run",
  "/jobs",
  "/v1",
  "/capabilities",
  "/recipes",
  "/weights",
  "/instances",
  "/downloads",
  "/plugins",
];

const GIQ = process.env.GIQ_URL ?? "http://127.0.0.1:8084";

export default defineConfig({
  base: "/dash/",
  // thirdPartyLicenses: THIRD_PARTY_LICENSES.txt into the build, and a failed
  // build on a bundled package outside the licence allowlist.
  plugins: [react(), ...thirdPartyLicenses()],
  build: {
    outDir: "../src/giq/static/ui",
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(
      API_PREFIXES.map((prefix) => [
        prefix,
        {
          target: GIQ,
          changeOrigin: true,
          // giq's access middleware refuses any foreign Origin (the dev
          // server's localhost:5173 is one), and the proxy is the page's own
          // backend here, so the header goes rather than being faked.
          configure: (proxy) => {
            proxy.on("proxyReq", (req) => req.removeHeader("origin"));
          },
        },
      ]),
    ),
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "plugins/*/src/**/*.test.ts"],
  },
});
