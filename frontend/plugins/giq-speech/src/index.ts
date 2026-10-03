// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { PluginModule } from "@giq/plugin-ui";
import { AsrTab } from "./AsrTab";
import { TtsTab } from "./TtsTab";
import { VoiceTab } from "./VoiceTab";

/* giq-speech's sandbox panels (ADR-004 D6), by the ids its manifest declares. */
export const panels: PluginModule["panels"] = { asr: AsrTab, tts: TtsTab, voice: VoiceTab };
