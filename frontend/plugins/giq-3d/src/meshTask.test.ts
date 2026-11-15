// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { NO_OPTIONS, optionFields } from "./meshTask";

describe("optionFields", () => {
  it("sends only the mode when the form sets nothing", () => {
    expect(optionFields(NO_OPTIONS)).toEqual({ mode: "game" });
  });

  it("sends what the form sets, as numbers", () => {
    expect(optionFields({ mode: "game", faces: "800", textureSize: "1024", close: "0", seed: "7" })).toEqual({
      mode: "game",
      faces: 800,
      texture_size: 1024,
      close: 0,
      seed: 7,
    });
  });

  it("leaves closing out of a full export", () => {
    expect(optionFields({ ...NO_OPTIONS, mode: "full", close: "4" })).toEqual({ mode: "full" });
  });

  it("drops what is not a whole number", () => {
    expect(optionFields({ ...NO_OPTIONS, faces: "1500.5", seed: "abc" })).toEqual({ mode: "game" });
  });
});
