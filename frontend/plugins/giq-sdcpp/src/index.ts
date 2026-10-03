// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { PluginModule } from "@giq/plugin-ui";
import { EditTab } from "./EditTab";
import { T2iTab } from "./T2iTab";

/* giq-sdcpp's sandbox panels (ADR-004 D6), by the ids its manifest declares. */
export const panels: PluginModule["panels"] = { t2i: T2iTab, edit: EditTab };
