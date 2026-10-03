// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { api } from "@giq/plugin-ui";
import type { JobRequest, JobStatusResponse } from "../../../src/api/types";

export interface ImageResult {
  b64: string;
  seed: number | null;
}

/* Image panels submit through the job API with ?wait=true, so the request
   holds until the image exists — no polling, and closing the tab while the
   job still waits in the queue cancels it. A task that failed reports its
   error in its result, not as an HTTP error. */
export async function runImageJob(req: JobRequest, signal: AbortSignal): Promise<ImageResult> {
  const d = await api.postJSON<JobStatusResponse>("/run?wait=true", req, { signal });
  const res = d.results?.[0] as { image_b64?: string; seed?: number; error?: string } | undefined;
  if (res?.error) throw new Error(res.error);
  if (!res?.image_b64) throw new Error(`job ${d.job_id} ${d.status} without an image`);
  return { b64: res.image_b64, seed: res.seed ?? null };
}
