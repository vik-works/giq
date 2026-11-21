// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  DASH,
  ellipsize,
  fmtAgo,
  fmtBytes,
  fmtDur,
  fmtEnergy,
  fmtGB,
  fmtNum,
  fmtPct,
  fmtSpan,
  fmtTemp,
  fmtTok,
  fmtUptime,
  fmtWatts,
} from "./format";

// Intl output uses no-break spaces between number and unit in some locales;
// normalising them keeps the expectations readable.
const n = (s: string) => s.replace(/[  ]/g, " ");

describe("format", () => {
  it("renders missing readings as a dash", () => {
    for (const fn of [fmtBytes, fmtDur, fmtTok, fmtWatts, fmtTemp]) {
      expect(fn(null, "en")).toBe(DASH);
    }
    expect(fmtGB(undefined, "de")).toBe(DASH);
    expect(fmtAgo(null, "en")).toBe(DASH);
  });

  it("groups and separates decimals per language", () => {
    expect(n(fmtGB(1234.56, "en"))).toBe("1,234.6 GB");
    expect(n(fmtGB(1234.56, "de"))).toBe("1.234,6 GB");
    expect(fmtNum(12345, "de")).toBe("12.345");
    // Four digits too, so a column of counts is grouped alike.
    expect(fmtNum(2118, "de")).toBe("2.118");
    expect(fmtTok(2118, "de")).toBe("2.118");
  });

  it("picks a byte unit by size", () => {
    expect(n(fmtBytes(2.5e12, "en"))).toBe("2.50 TB");
    expect(n(fmtBytes(3.21e9, "de"))).toBe("3,2 GB");
    expect(n(fmtBytes(4.5e6, "en"))).toMatch(/^[45] MB$/);
    expect(n(fmtBytes(1500, "en"))).toBe("2 kB");
  });

  it("formats durations in ms, s and min", () => {
    expect(n(fmtDur(850, "en"))).toBe("850 ms");
    expect(n(fmtDur(1234, "de"))).toMatch(/^1,2 Sek\.$/);
    expect(n(fmtDur(120_000, "en"))).toBe("2 min");
  });

  it("says how long ago", () => {
    const now = 1_000_000_000_000;
    expect(fmtAgo(now / 1000 - 5, "en", now)).toBe("5s ago");
    expect(n(fmtAgo(now / 1000 - 3 * 3600, "de", now))).toMatch(/^vor 3/);
    expect(fmtAgo(now / 1000 - 3 * 86_400, "en", now)).toMatch(/^3/);
  });

  it("compacts token counts", () => {
    expect(fmtTok(999, "en")).toBe("999");
    expect(fmtTok(12_345, "en")).toBe("12.3K");
    expect(n(fmtTok(1_250_000, "de"))).toBe("1,3 Mio.");
  });

  it("formats percent, watts and temperature", () => {
    expect(fmtPct(72, "en")).toBe("72%");
    expect(n(fmtPct(72, "de"))).toBe("72 %");
    expect(n(fmtWatts(305.4, "en"))).toBe("305 W");
    expect(n(fmtTemp(65.4, "en"))).toBe("65°C");
  });

  it("keeps the two largest uptime units", () => {
    const s = 6 * 86_400 + 4 * 3600 + 59;
    expect(fmtUptime(s, "en")).toMatch(/6.*4/);
    expect(fmtUptime(s, "en")).not.toMatch(/59/);
    expect(fmtUptime(0, "en")).toMatch(/0/);
  });

  it("shortens GPU names and long labels", () => {
    expect(ellipsize("abcdefghij", 5)).toBe("abcd…");
    expect(ellipsize("abc", 5)).toBe("abc");
  });
});

describe("energy and long spans", () => {
  it("writes watt-hours, kilowatt-hours from 1000", () => {
    expect(fmtEnergy(null, "en")).toBe(DASH);
    expect(fmtEnergy(0, "en")).toBe("0 Wh");
    expect(fmtEnergy(0.046, "en")).toBe("0.05 Wh");
    expect(fmtEnergy(4.25, "de")).toBe("4,3 Wh");
    expect(fmtEnergy(812.4, "en")).toBe("812 Wh");
    expect(fmtEnergy(1234, "de")).toBe("1,23 kWh");
    expect(fmtEnergy(123_456, "en")).toBe("123.5 kWh");
  });

  it("switches to hours past ninety minutes", () => {
    expect(fmtSpan(30_000, "en")).toBe(fmtDur(30_000, "en"));
    expect(n(fmtSpan(3_600_000, "en"))).toMatch(/^60 min/);
    expect(n(fmtSpan(2.5 * 3_600_000, "en"))).toBe("2.5 hr");
    expect(n(fmtSpan(1234 * 3_600_000, "de"))).toBe("1.234 Std.");
  });
});
