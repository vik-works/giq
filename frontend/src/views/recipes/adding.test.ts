// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { recipeEntry } from "../../api/testing";
import type { Download, PluginEntry } from "../../api/types";
import { addOffer, caveat, downloadOf, isActive, verdict } from "./adding";

const plugin = (over: Partial<PluginEntry>): PluginEntry => ({
  name: "p",
  package: "p",
  summary: "",
  engines: [],
  modalities: [],
  recipes: [],
  needs: "",
  curated: true,
  installed: false,
  status: null,
  install: "uv pip install p",
  ...over,
});

describe("addOffer", () => {
  const here = recipeEntry({ name: "here" });
  const fetchable = recipeEntry({ name: "b-fetch", installed: false, availability: "fetchable" });
  const manual = recipeEntry({ name: "a-manual", installed: false, availability: "manual" });
  const unfit = recipeEntry({ name: "huge", installed: false, availability: "unfit" });
  const ocr = recipeEntry({ name: "glm", installed: false, availability: "fetchable", modalities: ["ocr"] });

  it("offers what is not on this machine, fetchable first, by modality", () => {
    const offer = addOffer([here, manual, fetchable, unfit, ocr], []);
    expect(offer.groups.map((g) => g.modality)).toEqual(["llm", "ocr"]);
    expect(addOffer([ocr, fetchable], []).groups.map((g) => g.modality)).toEqual(["llm", "ocr"]);
    expect(offer.groups[0]!.offered.map((r) => r.name)).toEqual(["b-fetch", "a-manual"]);
    expect(offer.unfit.map((r) => r.name)).toEqual(["huge"]);
  });

  it("puts a missing plugin under the modality it adds, and an engine plugin apart", () => {
    const sdcpp = plugin({ name: "giq-sdcpp", modalities: ["text2image", "image_edit"] });
    const vllm = plugin({ name: "giq-vllm", engines: ["vllm"] });
    const installed = plugin({ name: "giq-ocr", modalities: ["ocr"], installed: true, install: null });
    const offer = addOffer([], [sdcpp, vllm, installed]);
    expect(offer.groups.map((g) => g.modality)).toEqual(["text2image", "image_edit"]);
    expect(offer.groups[1]!.plugins).toEqual([sdcpp]);
    expect(offer.enginePlugins).toEqual([vllm]);
  });

  it("filters by what the user typed", () => {
    const offer = addOffer([fetchable, ocr], [plugin({ name: "giq-depth", modalities: ["depth"] })], "glm");
    expect(offer.groups.map((g) => g.modality)).toEqual(["ocr"]);
  });
});

describe("downloads", () => {
  const d = (over: Partial<Download>): Download => ({
    id: "1",
    recipe: "x",
    state: "done",
    bytes_total: 1,
    bytes_done: 1,
    current: null,
    error: null,
    queued_at: 1,
    started_at: 1,
    finished_at: 2,
    ...over,
  });
  it("takes a recipe's latest fetch", () => {
    const latest = d({ id: "2", queued_at: 5, state: "running" });
    expect(downloadOf("x", [d({}), latest, d({ id: "3", recipe: "y", queued_at: 9 })])).toBe(latest);
    expect(isActive(latest)).toBe(true);
    expect(isActive(d({}))).toBe(false);
  });
});

describe("verdict", () => {
  it("is the first failing check, else the first warning", () => {
    const r = recipeEntry({
      checks: [
        { check: "weights", status: "warn", message: "not on disk" },
        { check: "card", status: "fail", message: "too large" },
      ],
    });
    expect(verdict(r)).toBe("too large");
    expect(verdict(recipeEntry({}))).toBeNull();
  });
});

describe("caveat", () => {
  const missing = { check: "weights", status: "warn" as const, message: "not on disk; giq can fetch them" };
  it("does not repeat what the fetchable tag says", () => {
    expect(caveat(recipeEntry({ availability: "fetchable", checks: [missing] }))).toBeNull();
    const pinned = { check: "card", status: "warn" as const, message: "no GPU seen" };
    expect(caveat(recipeEntry({ availability: "fetchable", checks: [missing, pinned] }))).toBe("no GPU seen");
  });
  it("says why a manual recipe needs a hand", () => {
    const place = { check: "weights", status: "fail" as const, message: "place it there" };
    expect(caveat(recipeEntry({ availability: "manual", checks: [place] }))).toBe("place it there");
  });
});
