// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

export type MeshMode = "game" | "full";

/** The form's choices; empty strings leave a field to the server's default. */
export interface MeshOptions {
  mode: MeshMode;
  faces: string;
  textureSize: string;
  close: string;
  seed: string;
}

export const NO_OPTIONS: MeshOptions = { mode: "game", faces: "", textureSize: "", close: "", seed: "" };

/* What the server takes for each export (giq_3d_child._mesh_child), shown as
   the inputs' bounds and placeholders so the form refuses what it would. */
export const FACES: Record<MeshMode, { fallback: number; min: number; max: number }> = {
  game: { fallback: 3000, min: 200, max: 50000 },
  full: { fallback: 100000, min: 5000, max: 1000000 },
};
export const TEXTURE_SIZES = [1024, 2048, 4096] as const;
export const TEXTURE_SIZE: Record<MeshMode, number> = { game: 2048, full: 4096 };
export const MAX_CLOSE = 8;

const whole = (v: string): number | null => {
  const n = Number(v);
  return v.trim() !== "" && Number.isInteger(n) ? n : null;
};

/** The task's optional fields: only what the form actually sets. */
export function optionFields(o: MeshOptions): Record<string, unknown> {
  const task: Record<string, unknown> = { mode: o.mode };
  const faces = whole(o.faces);
  if (faces != null) task.faces = faces;
  const size = whole(o.textureSize);
  if (size != null) task.texture_size = size;
  const seed = whole(o.seed);
  if (seed != null) task.seed = seed;
  // Closing openings is the cage's; the full export has no such step.
  const close = whole(o.close);
  if (o.mode === "game" && close != null) task.close = close;
  return task;
}
