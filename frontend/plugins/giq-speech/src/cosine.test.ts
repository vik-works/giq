// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { cosine, verdict } from "./cosine";

describe("cosine", () => {
  it("is the dot product for unit vectors and scale-free otherwise", () => {
    expect(cosine([1, 0], [0, 1])).toBe(0);
    expect(cosine([0.6, 0.8], [0.6, 0.8])).toBeCloseTo(1, 12);
    expect(cosine([3, 4], [6, 8])).toBeCloseTo(1, 12);
    expect(cosine([1, 0], [-1, 0])).toBe(-1);
  });
  it("refuses mismatched embeddings", () => {
    expect(() => cosine([1], [1, 2])).toThrow(/differ/);
  });
});

describe("verdict", () => {
  it("uses the 0.5 / 0.25 thresholds", () => {
    expect(verdict(0.72)).toBe("same");
    expect(verdict(0.5)).toBe("same");
    expect(verdict(0.49)).toBe("inconclusive");
    expect(verdict(0.25)).toBe("inconclusive");
    expect(verdict(0.2)).toBe("different");
    expect(verdict(-0.3)).toBe("different");
  });
});
