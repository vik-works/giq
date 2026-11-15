// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { PluginModule } from "@giq/plugin-ui";
import { MeshTab } from "./MeshTab";

/* giq-3d's sandbox panel (ADR-004 D6), by the id its manifest declares. */
export const panels: PluginModule["panels"] = { mesh: MeshTab };
