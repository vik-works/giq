// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { RecipeEntry, Residency, RecipeCard } from "./types";

/* A recipe entry for tests: an installed, on-demand LLM recipe that fits
   now, with any field overridden. Imported by tests only, so the app bundle
   never carries it. */

type Over = Omit<Partial<RecipeEntry>, "residency" | "card"> & {
  residency?: Partial<Residency>;
  card?: Partial<RecipeCard>;
};

export function recipeEntry(over: Over = {}): RecipeEntry {
  const { residency, card, ...rest } = over;
  return {
    name: "x",
    label: "",
    detail: "",
    modalities: ["llm"],
    engine: "llama.cpp",
    runtime: "llama.cpp",
    aliases: [],
    capabilities: ["chat"],
    vision: false,
    reasoning: null,
    vram_gb: 8,
    measured: true,
    needed_gb: 10,
    lanes: 4,
    max_batch: null,
    voices: [],
    installed: true,
    availability: "ready",
    checks: [],
    weights: [],
    fit: "fits_now",
    last_used: null,
    instance: null,
    ...rest,
    residency: {
      policy: "auto",
      source: "default",
      reason: null,
      default_resident: false,
      ...residency,
    },
    card: {
      device: null,
      source: "default",
      effective: null,
      index: null,
      name: null,
      ...card,
    },
  };
}
