// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { Fit, Modality, Policy, RecipeEntry, RecipesResponse } from "../../api/types";
import { TOOLS_MODEL } from "./constants";
import { isCoreTab, type TabInfo } from "./tabs";

/* What each panel can offer, derived from the shared recipe list on every
   refresh (the old page read it once at load, so a recipe loaded or pinned
   since stayed labelled with its stale fit until a reload). */

export interface ModelOption {
  /** The recipe name: what a request sends as `model`. */
  model: string;
  vram_gb: number;
  fits: Fit;
  policy: Policy;
  reasoning: RecipeEntry["reasoning"];
  /** Weights present on disk. */
  onDisk: boolean;
}

export interface SandboxModels {
  /** The recipes have answered; before that, nothing is "missing". */
  ready: boolean;
  chat: ModelOption[];
  vision: ModelOption[];
  /** The chat panel's starting choice. */
  chatDefault: string | null;
  /** The tool-calling recipe exists and can load. */
  toolsAvailable: boolean;
  /** Every recipe that can load for a modality: what a plugin's panel offers. */
  forModality: (modality: string) => ModelOption[];
}

const usable = (r: RecipeEntry) => r.fit !== "never";

/* A recipe can serve several modalities (ADR-003): flux_klein renders and
   edits from one process, so it belongs in both image tabs. */
export const serves = (r: RecipeEntry, modality: Modality | string): boolean =>
  (r.modalities as string[]).includes(modality);

function option(r: RecipeEntry): ModelOption {
  return {
    model: r.name,
    vram_gb: r.vram_gb,
    fits: r.fit,
    policy: r.residency.policy,
    reasoning: r.reasoning,
    onDisk: r.installed,
  };
}

export function sandboxModels(recipes: RecipesResponse | undefined): SandboxModels {
  if (!recipes) {
    return { ready: false, chat: [], vision: [], chatDefault: null, toolsAvailable: true, forModality: () => [] };
  }
  const usableRecipes = recipes.recipes.filter(usable);
  /* Every LLM that can load is selectable (the panel was once pinned to one
     model, so "test" on any other LLM card had nowhere to land). */
  const llms = usableRecipes.filter((r) => serves(r, "llm"));
  /* Default to whatever answers soonest: already loaded, else kept warm,
     else one resident by default. Alphabetical order (the fallback) picks an
     18 GB recipe nobody asked for. */
  const preferred =
    llms.find((r) => r.fit === "loaded") ??
    llms.find((r) => r.residency.policy === "pinned") ??
    llms.find((r) => r.residency.default_resident) ??
    llms[0];
  /* A plugin's panel: what is on disk first, then the rest (it says "not on
     disk" for those), each group by name. */
  const forModality = (modality: string) =>
    usableRecipes
      .filter((r) => serves(r, modality))
      .sort((a, b) => Number(b.installed) - Number(a.installed) || a.name.localeCompare(b.name))
      .map(option);
  return {
    ready: true,
    chat: llms.map(option),
    vision: usableRecipes.filter((r) => r.vision).map(option),
    chatDefault: preferred?.name ?? null,
    toolsAvailable: llms.some((r) => r.name === TOOLS_MODEL && r.residency.policy !== "off"),
    forModality,
  };
}

/** The options a tab offers in its model select. */
export function tabOptions(m: SandboxModels, tab: TabInfo): ModelOption[] {
  if (tab.id === "chat") return m.chat;
  if (tab.id === "vision") return m.vision;
  if (isCoreTab(tab.id)) return [];
  return m.forModality(tab.modality);
}

/** Tabs with nothing to run: greyed out in the tab bar once the catalog is known. */
export function disabledTabs(m: SandboxModels, tabs: TabInfo[]): Set<string> {
  const off = new Set<string>();
  if (!m.ready) return off;
  for (const tab of tabs) {
    if (tab.id === "tools" ? !m.toolsAvailable : !tabOptions(m, tab).length) off.add(tab.id);
  }
  return off;
}
