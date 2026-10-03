// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { Modality } from "../api/types";
import { registrationIndex } from "./modalities";

/* Series colours as CSS variable references, never resolved hex: an SVG
   fill of var(--s-llm) follows a theme switch by itself, where a colour read
   once with getComputedStyle would be stuck on the old theme. */

const KNOWN = new Set<string>([
  "llm",
  "text2image",
  "image_edit",
  "audio",
  "embed",
  "tts",
  "stt",
  "ocr",
  "depth",
]);

/** The fixed colour of a worker (its series slot). Colour follows the entity, not its rank.
 * A modality a plugin adds takes the series slot of its place in the server's
 * registration order (ADR-004 D6); past the eighth, the neutral "other". */
export function workerColor(worker: Modality | string): string {
  if (KNOWN.has(worker)) return `var(--s-${worker})`;
  return seriesColor(registrationIndex(worker));
}

/** Stack/legend order for workers: the slot order, so stacks look the same everywhere. */
export const WORKER_ORDER: readonly string[] = [
  "llm",
  "text2image",
  "audio",
  "embed",
  "image_edit",
  "tts",
  "stt",
  "ocr",
  "depth",
];

export const workerRank = (w: string): number => {
  const i = WORKER_ORDER.indexOf(w);
  return i < 0 ? WORKER_ORDER.length : i;
};

export const SERIES_SLOTS = 8;

/** Categorical slot i (0-based) as a CSS var; past the eighth, the neutral "other". */
export function seriesColor(i: number): string {
  return i >= 0 && i < SERIES_SLOTS ? `var(--series-${i + 1})` : "var(--series-other)";
}

/* Colours for an open-ended set of entities (models in the usage view).
   Assigned alphabetically, not by rank, so a model keeps its colour when a
   period or GPU filter changes what else is on screen; past eight they fold
   into the neutral "other". */
export function assignSeriesColors(keys: Iterable<string>): Map<string, string> {
  const out = new Map<string, string>();
  [...new Set(keys)].sort().forEach((k, i) => out.set(k, seriesColor(i)));
  return out;
}
