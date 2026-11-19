// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { api } from "@giq/plugin-ui";
import type { JobStatusResponse } from "../../../src/api/types";
/* giq-music's decoder: both panels play base64 audio from the job API, and
   both are built from this tree, so the one copy serves both. */
import { audioBlob } from "../../giq-music/src/audio";

/** What the sound was rendered with: the task's values, the recipe's, or the seed it drew. */
export interface SfxMeta {
  duration: number | null;
  seed: number | null;
  steps: number | null;
  guidance: number | null;
}

export interface SfxResult extends SfxMeta {
  audio: Blob;
  format: string;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/* Like the music panel: the job API with ?wait=true holds the request until
   the sound exists, and closing the tab while the job still waits in the
   queue cancels it. A task that failed reports its error in its result. */
export async function runSfxJob(
  model: string,
  task: Record<string, unknown>,
  signal: AbortSignal,
): Promise<SfxResult> {
  // text2sfx is this plugin's modality, not one of the dashboard's own.
  const req = { modality: "text2sfx", model, tasks: [task] };
  const d = await api.postJSON<JobStatusResponse>("/run?wait=true", req, { signal });
  const res = d.results?.[0] as Record<string, unknown> | undefined;
  if (typeof res?.error === "string" && res.error) throw new Error(res.error);
  if (typeof res?.audio_b64 !== "string" || !res.audio_b64) {
    throw new Error(`job ${d.job_id} ${d.status} without audio`);
  }
  const format = typeof res.format === "string" && res.format ? res.format : "flac";
  return {
    audio: audioBlob(res.audio_b64, format),
    format,
    duration: num(res.duration),
    seed: num(res.seed),
    steps: num(res.steps),
    guidance: num(res.guidance),
  };
}
