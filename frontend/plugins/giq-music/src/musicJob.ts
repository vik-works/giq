// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { api } from "@giq/plugin-ui";
import type { JobStatusResponse } from "../../../src/api/types";
import { audioBlob } from "./audio";

/** What the model settled on: the request's values, or the planner's. */
export interface MusicMeta {
  duration: number | null;
  bpm: number | null;
  keyScale: string | null;
  timeSignature: string | null;
  seed: number | null;
}

export interface MusicResult extends MusicMeta {
  audio: Blob;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/* Like the image panels: the job API with ?wait=true holds the request until
   the track exists, and closing the tab while the job still waits in the
   queue cancels it. A task that failed reports its error in its result. */
export async function runMusicJob(
  model: string,
  task: Record<string, unknown>,
  signal: AbortSignal,
): Promise<MusicResult> {
  // text2music is this plugin's modality, not one of the dashboard's own.
  const req = { modality: "text2music", model, tasks: [task] };
  const d = await api.postJSON<JobStatusResponse>("/run?wait=true", req, { signal });
  const res = d.results?.[0] as Record<string, unknown> | undefined;
  if (typeof res?.error === "string" && res.error) throw new Error(res.error);
  if (typeof res?.audio_b64 !== "string" || !res.audio_b64) {
    throw new Error(`job ${d.job_id} ${d.status} without audio`);
  }
  return {
    audio: audioBlob(res.audio_b64, str(res.format) ?? "mp3"),
    duration: num(res.duration),
    bpm: num(res.bpm),
    keyScale: str(res.key_scale),
    timeSignature: str(res.time_signature),
    seed: num(res.seed),
  };
}
