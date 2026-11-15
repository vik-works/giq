// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* A track built on audio of the user's own (ACE-Step): what it is for, and
   the task fields that say so. "reference" lends the track its sound; the
   others build on the audio, which is then the track's source. */

export type AudioUse = "reference" | "cover" | "lego" | "complete";

export const AUDIO_USES: AudioUse[] = ["reference", "cover", "lego", "complete"];

/** ACE-Step's stems: the track lego adds. */
export const TRACKS = [
  "guitar", "bass", "drums", "keyboard", "synth", "strings",
  "brass", "woodwinds", "percussion", "vocals", "backing_vocals", "fx",
] as const;

export interface AudioGuideValue {
  file: File | null;
  use: AudioUse;
  /** How closely a cover follows the audio, 0-1; empty leaves it to the model. */
  strength: string;
  track: (typeof TRACKS)[number];
}

export const NO_AUDIO: AudioGuideValue = { file: null, use: "cover", strength: "", track: "guitar" };

/** The audio is what the track is built from: its length is the track's, and the planner sits out. */
export const isSource = (v: AudioGuideValue): boolean => v.file != null && v.use !== "reference";

/** The task fields for ``v``, the file read as base64 by ``b64``. */
export async function audioFields(
  v: AudioGuideValue,
  b64: (f: Blob) => Promise<string>,
): Promise<Record<string, unknown>> {
  if (!v.file) return {};
  const data = await b64(v.file);
  if (v.use === "reference") return { reference_audio_b64: data };
  const fields: Record<string, unknown> = { source_audio_b64: data, mode: v.use };
  if (v.use === "cover" && v.strength !== "") fields.cover_strength = Number(v.strength);
  if (v.use === "lego") fields.track = v.track;
  return fields;
}
