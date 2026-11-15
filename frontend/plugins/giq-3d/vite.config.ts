// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import { giqPluginUi } from "../../plugin-ui/vite.ts";
import { thirdPartyLicenses } from "../../scripts/third-party-licenses.js";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

// Built into the dashboard's tree, so the wheel and the UI tarball carry it.
// Unlike the other panels this one bundles a package of its own (three.js),
// so it writes its own THIRD_PARTY_LICENSES.txt beside the module: the
// dashboard's file lists only what the dashboard's build pulls in.
const config = giqPluginUi({
  name: "giq-3d",
  root: here("."),
  outDir: here("../../../src/giq/static/ui/plugins/giq-3d"),
});

export default defineConfig({
  ...config,
  plugins: [
    ...(config.plugins ?? []),
    ...thirdPartyLicenses("the giq-3d dashboard panel (giq/static/ui/plugins/giq-3d/)"),
  ],
});
