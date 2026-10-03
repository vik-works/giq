// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* The recipe the text-to-speech panel speaks with when the sandbox offers
   none: kokoro, small (0.5 GB) and loaded on demand without evicting the
   resident set. */
export const TTS_FALLBACK = "kokoro";

/* /capabilities reports no voices for kokoro, so the list is the worker's
   own mapping: "alloy" is the OpenAI-style default and maps to af_heart. */
export const TTS_VOICES = ["alloy", "af_heart", "af_bella", "am_michael", "bf_emma"] as const;
export const TTS_DEFAULT_VOICE = "alloy";

/* ecapa-tdnn voiceprints: the same speaker typically scores ≥ 0.5 cosine,
   different speakers < 0.25; the band between is honestly inconclusive. */
export const VOICE_SAME_MIN = 0.5;
export const VOICE_INCONCLUSIVE_MIN = 0.25;
