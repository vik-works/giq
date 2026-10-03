// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { Download, PluginEntry, RecipeEntry } from "../../api/types";
import { primaryModality } from "../../lib/recipes";

/* What the Add page offers (ADR-005 D6), free of React so it is testable.
   "On this machine" is what has weights on disk; everything else is offered
   here, grouped by the modality it adds: recipes giq can fetch or the
   operator can place, and the curated plugins that would bring a modality
   this giq does not have. A recipe that cannot run on this machine at all
   is listed apart, with the reason, never mixed in with what can. */

export interface AddGroup {
  modality: string;
  offered: RecipeEntry[];
  plugins: PluginEntry[];
}

export interface AddOffer {
  groups: AddGroup[];
  /** Recipes that cannot run here as the machine is. */
  unfit: RecipeEntry[];
  /** Plugins not installed that add no modality of their own (an engine). */
  enginePlugins: PluginEntry[];
}

/** Whether a recipe belongs on "On this machine" rather than the Add page. */
export const onThisMachine = (r: RecipeEntry): boolean => r.installed;

function matches(r: RecipeEntry, query: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  return [r.name, r.label, r.detail, r.engine, ...r.modalities].some((s) => s.toLowerCase().includes(q));
}

function pluginMatches(p: PluginEntry, query: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  return [p.name, p.summary, ...p.modalities, ...p.engines, ...p.recipes].some((s) =>
    s.toLowerCase().includes(q),
  );
}

/* The order the dashboard lists modalities in everywhere: LLMs first, then
   images, speech, documents; a plugin's own modality after them. */
const MODALITY_ORDER = ["llm", "text2image", "image_edit", "audio", "stt", "tts", "embed", "ocr", "depth"];
const rank = (m: string) => {
  const i = MODALITY_ORDER.indexOf(m);
  return i < 0 ? MODALITY_ORDER.length : i;
};

export function addOffer(
  recipes: RecipeEntry[],
  plugins: PluginEntry[],
  query = "",
): AddOffer {
  const missing = recipes.filter((r) => !onThisMachine(r) && matches(r, query));
  const absentPlugins = plugins.filter((p) => !p.installed && !p.status && pluginMatches(p, query));
  const byModality = new Map<string, AddGroup>();
  const group = (modality: string): AddGroup => {
    let g = byModality.get(modality);
    if (!g) {
      g = { modality, offered: [], plugins: [] };
      byModality.set(modality, g);
    }
    return g;
  };
  for (const r of missing) {
    if (r.availability !== "unfit") group(primaryModality(r)).offered.push(r);
  }
  for (const p of absentPlugins) for (const m of p.modalities) group(m).plugins.push(p);
  const order = (r: RecipeEntry) => (r.availability === "fetchable" ? 0 : 1);
  const groups = [...byModality.values()]
    .map((g) => ({ ...g, offered: [...g.offered].sort((a, b) => order(a) - order(b) || a.name.localeCompare(b.name)) }))
    .sort((a, b) => rank(a.modality) - rank(b.modality) || a.modality.localeCompare(b.modality));
  return {
    groups,
    unfit: missing.filter((r) => r.availability === "unfit"),
    enginePlugins: absentPlugins.filter((p) => p.modalities.length === 0),
  };
}

/** The recipe's latest fetch, if it has had one since the service started. */
export function downloadOf(name: string, downloads: Download[] | undefined): Download | undefined {
  return (downloads ?? []).filter((d) => d.recipe === name).sort((a, b) => b.queued_at - a.queued_at)[0];
}

export const isActive = (d: Download | undefined): boolean =>
  d != null && (d.state === "queued" || d.state === "running");

/** The first check that is not ok: the one line that says what stands in the way. */
export function verdict(r: RecipeEntry): string | null {
  return (
    r.checks.find((c) => c.status === "fail")?.message ??
    r.checks.find((c) => c.status === "warn")?.message ??
    null
  );
}

/** What is worth saying about a recipe on the Add page beyond its tag: a
 * fetchable recipe's missing weights are what the tag already says. */
export function caveat(r: RecipeEntry): string | null {
  const notes = r.checks.filter((c) => c.status !== "ok" && !(r.availability === "fetchable" && c.check === "weights"));
  return (notes.find((c) => c.status === "fail") ?? notes[0])?.message ?? null;
}
