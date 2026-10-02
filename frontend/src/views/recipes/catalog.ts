// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { EnginesResponse, Fit, Modality, RecipeEntry, WeightsItem } from "../../api/types";
import type { TagTone } from "../../components/Tag";
import { SANDBOX_TAB_FOR_WORKER } from "../../lib/sandboxLink";

/* The Recipes view's pure logic: the facets, what a card offers, and what a
   recipe's weights come to. Kept out of the components so it is testable
   without a DOM. */

export interface Filters {
  modality: ReadonlySet<string>;
  engine: ReadonlySet<string>;
}

/* Nothing selected means everything: an empty filter is not an empty result.
   A recipe matches a modality it serves, not only its first. */
export function matches(r: RecipeEntry, f: Filters): boolean {
  return (
    (!f.modality.size || r.modalities.some((m) => f.modality.has(m))) &&
    (!f.engine.size || f.engine.has(r.engine))
  );
}

/** Facet values with their counts over the whole catalog, sorted by name; a recipe counts once per value it has. */
export function facetCounts(recipes: RecipeEntry[], pick: (r: RecipeEntry) => string[]): [string, number][] {
  const n = new Map<string, number>();
  for (const r of recipes) for (const v of pick(r)) n.set(v, (n.get(v) ?? 0) + 1);
  return [...n.entries()].sort(([a], [b]) => a.localeCompare(b));
}

export function toggled<T>(set: ReadonlySet<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

/** A recipe's weights as /weights reports them: their total size, and the other recipes that share any. */
export function weightsOf(r: RecipeEntry, weights: WeightsItem[] | undefined): { bytes: number; sharedWith: string[] } {
  const mine = (weights ?? []).filter((w) => r.weights.includes(w.id));
  return {
    bytes: mine.reduce((a, w) => a + w.size_bytes, 0),
    sharedWith: [...new Set(mine.flatMap((w) => w.recipes))].filter((n) => n !== r.name).sort(),
  };
}

/* Fit badge tones. "fits" is the ordinary case and stays neutral; only the
   answers that change what happens next (a load evicts something, or it
   cannot load) carry a status colour. */
export const FIT_TONE: Record<Fit, TagTone> = {
  loaded: "good",
  fits_now: "neutral",
  fits_after_eviction: "serious",
  wont_fit_now: "critical",
  never: "critical",
};

/** Why "Test in sandbox" is unavailable, as a recipes.json key; null when it is available. */
export function sandboxBlock(r: RecipeEntry): string | null {
  if (!r.modalities.some((m: Modality) => SANDBOX_TAB_FOR_WORKER[m])) return "menu.noPanel";
  if (!r.installed) return "menu.nothingToRun";
  if (r.fit === "never") return "menu.tooLarge";
  return null;
}

/* Keeping a recipe warm that cannot fit its card, or has no weights to load,
   would only make the resident set unsatisfiable; switching it off is always
   allowed. Returns the recipes.json key of the reason, or null. */
export function pinBlock(r: RecipeEntry): string | null {
  if (r.residency.policy === "pinned") return null;
  if (r.fit === "never") return "residency.tooLarge";
  if (!r.installed) return "residency.noWeights";
  return null;
}

/* The runtime whose build is worth showing beside the engine name. "Which
   llama.cpp" changes behaviour and is easy to lose track of; for an
   interpreter-hosted backend the Python version is not the library version,
   and showing it would read as if it were. */
export const BUILT_ENGINES: ReadonlySet<string> = new Set(["llama.cpp", "sd.cpp"]);

/** runtime → its build string (prefix noise trimmed), or null when the binary is missing. */
export function engineVersions(e: EnginesResponse | undefined): Map<string, string | null> {
  return new Map(
    (e?.engines ?? []).map((x) => [
      x.name,
      x.present
        ? (x.version ?? "").replace(/^version:\s*/i, "").replace(/^stable-diffusion\.cpp\s*/i, "")
        : null,
    ]),
  );
}

/* Engine explanations live in recipes.json under engineNote.<key>; backend
   names contain dots, which i18next reads as nesting, so they map to keys. */
export const ENGINE_NOTE_KEY: Record<string, string> = {
  "llama.cpp": "llamaCpp",
  "sd.cpp": "sdCpp",
  "faster-whisper+pyannote": "fasterWhisperPyannote",
  speechbrain: "speechbrain",
  "faster-whisper": "fasterWhisper",
  kokoro: "kokoro",
  transformers: "transformers",
};
