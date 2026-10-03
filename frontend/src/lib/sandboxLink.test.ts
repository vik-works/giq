// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { parseSandboxHash, sandboxHref } from "./sandboxLink";
import { parseHash } from "./useHashRoute";

describe("sandboxHref", () => {
  it("routes to the tab and carries the model", () => {
    const href = sandboxHref("chat", "qwen3.8-27b");
    expect(href).toBe("#/sandbox/chat?model=qwen3.8-27b");
    expect(parseHash(href)).toEqual({ view: "sandbox", sub: "chat" });
    expect(parseSandboxHash(href)).toEqual({ tab: "chat", model: "qwen3.8-27b" });
    expect(parseSandboxHash(sandboxHref("t2i", "a b&c")).model).toBe("a b&c");
  });
});

describe("parseSandboxHash", () => {
  it("reads tab and model", () => {
    expect(parseSandboxHash("#/sandbox/t2i?model=zimage")).toEqual({ tab: "t2i", model: "zimage" });
    expect(parseSandboxHash("#/sandbox/chat")).toEqual({ tab: "chat", model: null });
  });
  it("is tolerant of shape", () => {
    expect(parseSandboxHash("#/sandbox")).toEqual({ tab: null, model: null });
    expect(parseSandboxHash("#/sandbox?model=gemma-4-12b")).toEqual({ tab: null, model: "gemma-4-12b" });
    expect(parseSandboxHash("#/sandbox/chat/?x=1&model=qwen3.6-27b")).toEqual({
      tab: "chat",
      model: "qwen3.6-27b",
    });
    // Plugins add panels, so the parser does not judge the id.
    expect(parseSandboxHash("#/sandbox/nope?model=")).toEqual({ tab: "nope", model: null });
    expect(parseSandboxHash("#/sandbox/%E0%A4%A?model=a%20b")).toEqual({ tab: null, model: "a b" });
  });
});

describe("tabForModality", () => {
  it("sends LLMs to chat and other modalities to the plugin panel declared for them", async () => {
    const { setPluginUis } = await import("../plugins/panels");
    const { tabForModality } = await import("./sandboxLink");
    setPluginUis([
      {
        plugin: "giq-sdcpp",
        base: "/plugins/giq-sdcpp/ui/",
        api_version: 1,
        module: "index.js",
        panels: [{ id: "t2i", modality: "text2image", label: "tab.t2i" }],
      },
    ]);
    expect(tabForModality("llm")).toBe("chat");
    expect(tabForModality("text2image")).toBe("t2i");
    expect(tabForModality("ocr")).toBeNull();
    setPluginUis([]);
  });
});
