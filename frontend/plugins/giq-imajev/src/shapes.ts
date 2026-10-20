// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* Pure shapes behind the decide result cards: verdicts, orderings and
   widths, so the components stay presentational. */

export interface DecideAnswer {
  type: string;
  choice?: string | null;
  noul?: number | null;
  score?: number | null;
  labels?: string[] | null;
  probabilities?: Record<string, number>;
  legend?: Record<string, string> | null;
  threshold?: number | null;
  confidence?: number | null;
  unknown_probability?: number;
  unknown_probabilities?: Record<string, number> | null;
  abstained?: boolean;
}

/** P(yes) at or above one half reads Yes; below it reads No. */
export function noulVerdict(noul: number | null | undefined): "yes" | "no" | null {
  if (noul == null) return null;
  return noul >= 0.5 ? "yes" : "no";
}

/** Label probabilities, most probable first (choice and multi). */
export function sortedProbs(a: DecideAnswer): [string, number][] {
  return Object.entries(a.probabilities ?? {}).sort(([, x], [, y]) => y - x);
}

export interface ScoreLevel {
  level: string;
  description?: string;
  p: number;
}

/* Score levels in scale order: the bars walk the scale low to high, each
   step with the legend's wording, so the shape reads as a distribution. */
export function scoreLevels(a: DecideAnswer): ScoreLevel[] {
  const probs = a.probabilities ?? {};
  const keys = Object.keys(a.legend ?? {});
  const order = (keys.length ? keys : Object.keys(probs)).sort((x, y) => Number(x) - Number(y));
  return order.map((level) => ({
    level,
    description: a.legend?.[level],
    p: probs[level] ?? 0,
  }));
}

