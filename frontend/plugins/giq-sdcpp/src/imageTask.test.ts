// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { NO_OPTIONS, optionFields } from "./imageTask";

describe("optionFields", () => {
  it("sends nothing when every option is left empty", () => {
    expect(optionFields(NO_OPTIONS)).toEqual({});
  });

  it("sends what was given, trimmed", () => {
    const v = { seed: "42", negative: " blurry ", width: "768", height: "1024" };
    expect(optionFields(v)).toEqual({ seed: 42, negative_prompt: "blurry", width: 768, height: 1024 });
  });

  it("sends a size only as a pair", () => {
    expect(optionFields({ ...NO_OPTIONS, width: "768" })).toEqual({});
    expect(optionFields({ ...NO_OPTIONS, height: "768" })).toEqual({});
  });

  it("drops values that are not whole numbers", () => {
    expect(optionFields({ seed: "-1", negative: "", width: "7.5", height: "x" })).toEqual({});
    expect(optionFields({ ...NO_OPTIONS, width: "0", height: "512" })).toEqual({});
  });
});
