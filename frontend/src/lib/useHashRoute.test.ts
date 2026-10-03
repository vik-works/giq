// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { hrefFor, parseHash } from "./useHashRoute";

describe("parseHash", () => {
  it("maps the dashboard's URLs", () => {
    expect(parseHash("")).toEqual({ view: "overview", sub: null });
    expect(parseHash("#/")).toEqual({ view: "overview", sub: null });
    expect(parseHash("#/usage")).toEqual({ view: "usage", sub: null });
    expect(parseHash("#/recipes")).toEqual({ view: "recipes", sub: null });
    expect(parseHash("#/recipes/add")).toEqual({ view: "recipes", sub: "add" });
    expect(parseHash("#/recipes/other")).toEqual({ view: "recipes", sub: null });
    expect(parseHash("#/inventory?recipe=x")).toEqual({ view: "inventory", sub: null });
    // The address before ADR-003 still lands on the Recipes view.
    expect(parseHash("#/models")).toEqual({ view: "recipes", sub: null });
    expect(parseHash("#/sandbox")).toEqual({ view: "sandbox", sub: null });
    expect(parseHash("#/sandbox/chat")).toEqual({ view: "sandbox", sub: "chat" });
  });
  it("sends anything unknown to the overview", () => {
    expect(parseHash("#/nope")).toEqual({ view: "overview", sub: null });
    expect(parseHash("#garbage")).toEqual({ view: "overview", sub: null });
  });
  it("round-trips through hrefFor", () => {
    expect(hrefFor("overview")).toBe("#/");
    expect(parseHash(hrefFor("sandbox", "t2i"))).toEqual({ view: "sandbox", sub: "t2i" });
  });
});
