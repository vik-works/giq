// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import type { UserConfig } from "vite";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export interface PluginUiOptions {
  /** The plugin's name, as it registers (giq-sdcpp). */
  name: string;
  /** The plugin UI's directory: src/index.ts(x), public/manifest.json, public/locales/. */
  root: string;
  /** Where the build goes; the plugin's `ui` directory. */
  outDir: string;
}

/* The Vite preset for a plugin's dashboard UI (ADR-004 D6): one ES module,
   one stylesheet, and public/ (the manifest and the strings) copied beside
   them. React, the JSX runtime, react-i18next and @giq/plugin-ui resolve to
   the dashboard's host object rather than into the bundle. */
export function giqPluginUi({ name, root, outDir }: PluginUiOptions): UserConfig {
  return {
    root,
    publicDir: `${root}/public`,
    plugins: [react()],
    define: { "process.env.NODE_ENV": JSON.stringify("production") },
    resolve: {
      alias: [
        { find: /^react\/jsx-runtime$/, replacement: here("./shims/jsx-runtime.ts") },
        { find: /^react\/jsx-dev-runtime$/, replacement: here("./shims/jsx-runtime.ts") },
        { find: /^react$/, replacement: here("./shims/react.ts") },
        { find: /^react-i18next$/, replacement: here("./shims/react-i18next.ts") },
        { find: /^@giq\/plugin-ui$/, replacement: here("./index.ts") },
      ],
    },
    build: {
      outDir,
      emptyOutDir: true,
      copyPublicDir: true,
      lib: {
        entry: `${root}/src/index.ts`,
        formats: ["es"],
        fileName: () => "index.js",
        cssFileName: "index",
      },
    },
    logLevel: "warn",
    // A plugin's name keeps two builds' caches apart.
    cacheDir: `node_modules/.vite-plugin-${name}`,
  };
}
