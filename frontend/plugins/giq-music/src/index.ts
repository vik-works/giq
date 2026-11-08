// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { PluginModule } from "@giq/plugin-ui";
import { MusicTab } from "./MusicTab";
import { SoundTab } from "./SoundTab";

/* giq-music's sandbox panels (ADR-004 D6), by the id its manifest declares. */
export const panels: PluginModule["panels"] = { music: MusicTab, sound: SoundTab };
