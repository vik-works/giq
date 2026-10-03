// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { VOICE_INCONCLUSIVE_MIN, VOICE_SAME_MIN } from "./constants";

/* Cosine similarity of two voiceprints. giq returns them L2-normalised, where
   this is the plain dot product; dividing by the norms keeps it right for an
   embedding that is not. Mismatched lengths are a server bug, not a score. */
export function cosine(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length || !a.length) throw new Error(`embedding sizes differ: ${a.length} vs ${b.length}`);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}

export type Verdict = "same" | "inconclusive" | "different";

export const verdict = (cos: number): Verdict =>
  cos >= VOICE_SAME_MIN ? "same" : cos >= VOICE_INCONCLUSIVE_MIN ? "inconclusive" : "different";
