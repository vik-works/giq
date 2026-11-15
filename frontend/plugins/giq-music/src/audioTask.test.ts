// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { audioFields, isSource, NO_AUDIO, type AudioGuideValue } from "./audioTask";

const file = new File(["x"], "beat.wav", { type: "audio/wav" });
const b64 = async () => "QUJD";
const with_ = (v: Partial<AudioGuideValue>): AudioGuideValue => ({ ...NO_AUDIO, file, ...v });

describe("audioFields", () => {
  it("sends nothing without a file", async () => {
    expect(await audioFields(NO_AUDIO, b64)).toEqual({});
    expect(isSource(NO_AUDIO)).toBe(false);
  });

  it("sends a reference as a reference, the planner kept", async () => {
    const v = with_({ use: "reference" });
    expect(await audioFields(v, b64)).toEqual({ reference_audio_b64: "QUJD" });
    expect(isSource(v)).toBe(false);
  });

  it("sends a cover with its strength when one is given", async () => {
    expect(await audioFields(with_({ use: "cover" }), b64)).toEqual({ source_audio_b64: "QUJD", mode: "cover" });
    expect(await audioFields(with_({ use: "cover", strength: "0.4" }), b64)).toEqual({
      source_audio_b64: "QUJD",
      mode: "cover",
      cover_strength: 0.4,
    });
  });

  it("sends lego with its track, and never a strength", async () => {
    const v = with_({ use: "lego", track: "bass", strength: "0.4" });
    expect(await audioFields(v, b64)).toEqual({ source_audio_b64: "QUJD", mode: "lego", track: "bass" });
    expect(isSource(v)).toBe(true);
  });
});
