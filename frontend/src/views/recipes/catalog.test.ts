// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { recipeEntry as recipe } from "../../api/testing";
import type { WeightsItem } from "../../api/types";
import { engineVersions, facetCounts, matches, pinBlock, sandboxBlock, weightsOf } from "./catalog";

describe("filters", () => {
  const rs = [
    recipe({ name: "a" }),
    recipe({ name: "b", modalities: ["tts"], engine: "kokoro" }),
    recipe({ name: "c", modalities: ["text2image", "image_edit"], engine: "sd.cpp" }),
  ];
  it("counts facets over the whole catalog, a recipe once per modality it serves", () => {
    expect(facetCounts(rs, (r) => r.modalities)).toEqual([
      ["image_edit", 1],
      ["llm", 1],
      ["text2image", 1],
      ["tts", 1],
    ]);
  });
  it("treats an empty filter as everything, and matches any modality a recipe serves", () => {
    const none = { modality: new Set<string>(), engine: new Set<string>() };
    expect(rs.filter((r) => matches(r, none))).toHaveLength(3);
    expect(rs.filter((r) => matches(r, { ...none, engine: new Set(["kokoro"]) })).map((r) => r.name)).toEqual(["b"]);
    expect(rs.filter((r) => matches(r, { ...none, modality: new Set(["image_edit"]) })).map((r) => r.name)).toEqual([
      "c",
    ]);
  });
});

describe("weights", () => {
  const w = (over: Partial<WeightsItem>): WeightsItem => ({ size_bytes: 0, recipes: [], ...over }) as WeightsItem;
  it("sums a recipe's checkpoints and names who shares them", () => {
    const items = [
      w({ id: "a", size_bytes: 5, recipes: ["nvfp4", "nvfp4-chat"] }),
      w({ id: "b", size_bytes: 7, recipes: ["other"] }),
    ];
    expect(weightsOf(recipe({ name: "nvfp4", weights: ["a"] }), items)).toEqual({
      bytes: 5,
      sharedWith: ["nvfp4-chat"],
    });
    expect(weightsOf(recipe({ name: "x", weights: [] }), undefined)).toEqual({ bytes: 0, sharedWith: [] });
  });
});

describe("what a card offers", () => {
  it("blocks keeping warm what can never fit or is not installed, unless it already is", () => {
    expect(pinBlock(recipe({ fit: "never" }))).toBe("residency.tooLarge");
    expect(pinBlock(recipe({ installed: false }))).toBe("residency.noWeights");
    expect(pinBlock(recipe({ residency: { policy: "pinned" }, fit: "never" }))).toBeNull();
  });
  it("offers the sandbox only with a panel, weights and a card it fits", () => {
    expect(sandboxBlock(recipe({ modalities: ["ocr"] }))).toBe("menu.noPanel");
    expect(sandboxBlock(recipe({ installed: false }))).toBe("menu.nothingToRun");
    expect(sandboxBlock(recipe({ fit: "never" }))).toBe("menu.tooLarge");
    expect(sandboxBlock(recipe({}))).toBeNull();
  });
});

describe("engines", () => {
  it("trims engine build strings and marks a missing binary", () => {
    const v = engineVersions({
      engines: [
        { name: "llama.cpp", binary: "x", present: true, version: "version: 0.2.0 (build 1)", error: null },
        { name: "sd.cpp", binary: "y", present: true, version: "stable-diffusion.cpp commit 2251699", error: null },
        { name: "retired-engine", binary: null, present: false, version: null, error: "gone" },
      ],
    });
    expect(v.get("llama.cpp")).toBe("0.2.0 (build 1)");
    expect(v.get("sd.cpp")).toBe("commit 2251699");
    expect(v.get("retired-engine")).toBeNull();
  });
});
