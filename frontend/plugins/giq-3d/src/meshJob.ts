// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { api } from "@giq/plugin-ui";
import type { JobStatusResponse } from "../../../src/api/types";

export interface MeshResult {
  glb: Blob;
  faces: number | null;
  textureSize: number | null;
  seed: number | null;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** A base64 GLB as a Blob the viewer and the download link can both take. */
export function glbBlob(b64: string): Blob {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Blob([bytes], { type: "model/gltf-binary" });
}

/* Like the image panels: the job API with ?wait=true holds the request until
   the asset exists, and closing the tab while the job still waits in the
   queue cancels it. A task that failed reports its error in its result. */
export async function runMeshJob(
  model: string,
  task: Record<string, unknown>,
  signal: AbortSignal,
): Promise<MeshResult> {
  // image2mesh is this plugin's modality, not one of the dashboard's own.
  const req = { modality: "image2mesh", model, tasks: [task] };
  const d = await api.postJSON<JobStatusResponse>("/run?wait=true", req, { signal });
  const res = d.results?.[0] as Record<string, unknown> | undefined;
  if (typeof res?.error === "string" && res.error) throw new Error(res.error);
  if (typeof res?.glb_b64 !== "string" || !res.glb_b64) {
    throw new Error(`job ${d.job_id} ${d.status} without a mesh`);
  }
  return {
    glb: glbBlob(res.glb_b64),
    faces: num(res.faces),
    textureSize: num(res.texture_size),
    seed: num(res.seed),
  };
}
