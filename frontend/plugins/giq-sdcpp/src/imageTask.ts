// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* The optional fields both image panels share, as typed in the form. Empty
   means "the server's default": a random seed, the configured size. */
export interface ImageOptionsValue {
  seed: string;
  negative: string;
  width: string;
  height: string;
}

export const NO_OPTIONS: ImageOptionsValue = { seed: "", negative: "", width: "", height: "" };

const whole = (s: string, min: number): number | null => {
  const n = Number(s.trim());
  return s.trim() !== "" && Number.isInteger(n) && n >= min ? n : null;
};

/* The task fields the form's options add. A size goes as a pair or not at
   all: the server ignores a lone side, so sending one would only mislead.
   The server also snaps the pair and scales it into the recipe's budget,
   keeping the aspect, so any whole size is worth sending. */
export function optionFields(v: ImageOptionsValue): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const seed = whole(v.seed, 0);
  if (seed != null) out.seed = seed;
  if (v.negative.trim()) out.negative_prompt = v.negative.trim();
  const width = whole(v.width, 1);
  const height = whole(v.height, 1);
  if (width != null && height != null) Object.assign(out, { width, height });
  return out;
}

/* An image file's pixel size, or null when the browser cannot decode it. */
export async function imageSize(file: Blob): Promise<{ width: number; height: number } | null> {
  try {
    const bitmap = await createImageBitmap(file);
    const size = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return size;
  } catch {
    return null;
  }
}
