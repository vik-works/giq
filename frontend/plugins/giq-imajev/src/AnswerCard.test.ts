// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* Pure shapes behind the decide result cards: verdicts, orderings and
   widths, so the components stay presentational. */

import { describe, expect, it } from "vitest";
import { noulVerdict, scoreLevels, sortedProbs } from "./shapes";

describe("noulVerdict", () => {
  it("reads P(yes) at or above one half as Yes", () => {
    expect(noulVerdict(0.5)).toBe("yes");
    expect(noulVerdict(0.75)).toBe("yes");
  });

  it("reads below one half as No, and nothing as no verdict", () => {
    expect(noulVerdict(0.49)).toBe("no");
    expect(noulVerdict(null)).toBeNull();
    expect(noulVerdict(undefined)).toBeNull();
  });
});

describe("sortedProbs", () => {
  it("orders labels most probable first", () => {
    expect(
      sortedProbs({ type: "choice", probabilities: { billing: 0.11, shipping: 0.89 } }).map(([l]) => l),
    ).toEqual(["shipping", "billing"]);
  });
});

describe("scoreLevels", () => {
  it("walks the scale low to high with the legend's wording", () => {
    const levels = scoreLevels({
      type: "score",
      legend: { "0": "bad", "1": "ok", "2": "great" },
      probabilities: { "2": 0.1, "0": 0.7, "1": 0.2 },
    });
    expect(levels.map((l) => l.level)).toEqual(["0", "1", "2"]);
    expect(levels[0]).toMatchObject({ description: "bad", p: 0.7 });
  });

  it("falls back to the probability keys when no legend shipped", () => {
    expect(scoreLevels({ type: "score", probabilities: { "1": 0.6 } })).toEqual([
      { level: "1", description: undefined, p: 0.6 },
    ]);
  });
});
