// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import { giqPluginUi } from "../../plugin-ui/vite.ts";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

// Built into the dashboard's tree, so the wheel and the UI tarball carry it.
export default defineConfig(
  giqPluginUi({
    name: "giq-sdcpp",
    root: here("."),
    outDir: here("../../../src/giq/static/ui/plugins/giq-sdcpp"),
  }),
);
