// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { AppEntry, GpuEra, StatsUsage, UsageAppRow, UsageEnergy } from "../../api/types";
import {
  ANONYMOUS,
  appFilterOptions,
  appFilterValue,
  appFromHash,
  bucketDate,
  bucketLabel,
  buildUsageChart,
  cardNameFor,
  energySplit,
  rollupApps,
  UNKNOWN,
  eraTokens,
  erasNewestFirst,
  isCurrentEra,
  predatesTokenAccounting,
  usageDomain,
} from "./usageModel";

const NOW = new Date(2026, 8, 28, 14, 30); // Sep 28 2026, 14:30 local

describe("usageDomain", () => {
  it("covers every hour of today up to now", () => {
    const d = usageDomain("day", [], NOW);
    expect(d).toHaveLength(15);
    expect(d[0]).toBe("2026-09-28 00:00");
    expect(d.at(-1)).toBe("2026-09-28 14:00");
  });
  it("covers the last 7 and 30 calendar days, ending today", () => {
    const w = usageDomain("week", [], NOW);
    expect(w).toEqual([
      "2026-09-22",
      "2026-09-23",
      "2026-09-24",
      "2026-09-25",
      "2026-09-26",
      "2026-09-27",
      "2026-09-28",
    ]);
    const m = usageDomain("month", [], NOW);
    expect(m).toHaveLength(30);
    expect(m[0]).toBe("2026-08-30");
  });
  it("uses only the months seen for all time, sorted and unique", () => {
    expect(usageDomain("all", ["2026-09", "2026-07", "2026-09"], NOW)).toEqual(["2026-07", "2026-09"]);
  });
});

describe("bucket labels", () => {
  it("parses every key shape as local time", () => {
    expect(bucketDate("2026-09-28 14:00")).toEqual(new Date(2026, 8, 28, 14));
    expect(bucketDate("2026-09-28")).toEqual(new Date(2026, 8, 28));
    expect(bucketDate("2026-09")).toEqual(new Date(2026, 8, 1));
  });
  it("formats per language", () => {
    expect(bucketLabel("2026-09-28 14:00", "day", "de")).toBe("14:00");
    expect(bucketLabel("2026-09-28", "week", "en")).toBe("Sep 28");
    expect(bucketLabel("2026-09-28", "week", "de")).toBe("28. Sept.");
    expect(bucketLabel("2026-09", "all", "en")).toBe("Sep 2026");
  });
});

const usage = (over: Partial<StatsUsage> = {}): StatsUsage => ({
  period: "week",
  gpu: null,
  app: null,
  since: 0,
  until: null,
  totals: { jobs: 0, failed: 0, tokens_in: 0, tokens_out: 0, gpu_ms: 0, energy_wh: 0 },
  recipes: [],
  series: [],
  apps: [],
  energy: { attributed_wh: 0, idle_wh: 0, estimated: false, cards: [] },
  ...over,
});

describe("buildUsageChart", () => {
  const d = usage({
    recipes: [
      { modality: "llm", recipe: "zeta", jobs: 3, failed: 0, tasks: 3, tokens_in: 900, tokens_out: 100, last_ts: 1 },
      { modality: "llm", recipe: "alpha", jobs: 1, failed: 0, tasks: 1, tokens_in: 10, tokens_out: 5, last_ts: 1 },
      { modality: "text2image", recipe: "flux", jobs: 2, failed: 0, tasks: 2, tokens_in: null, tokens_out: null, last_ts: 1 },
    ],
    series: [
      { b: "2026-09-27", modality: "llm", recipe: "zeta", jobs: 3, tokens_in: 900, tokens_out: 100 },
      { b: "2026-09-27", modality: "llm", recipe: "alpha", jobs: 1, tokens_in: 10, tokens_out: 5 },
      { b: "2026-09-27", modality: "text2image", recipe: "flux", jobs: 2, tokens_in: null, tokens_out: null },
    ],
  });
  const c = buildUsageChart(d, "week", NOW);

  it("spans the whole domain, empty buckets included", () => {
    expect(c.buckets).toHaveLength(7);
    expect(c.buckets[0]?.segments).toEqual([]);
  });
  it("stacks only models with tokens, in the table's order", () => {
    expect(c.models.map((m) => m.recipe)).toEqual(["zeta", "alpha"]);
    expect(c.buckets[5]?.segments.map((s) => [s.id, s.value])).toEqual([
      ["llm/zeta", 1000],
      ["llm/alpha", 15],
    ]);
  });
  it("assigns colours alphabetically over every model in the window", () => {
    expect(c.colors.get("llm/alpha")).toBe("var(--series-1)");
    expect(c.colors.get("llm/zeta")).toBe("var(--series-2)");
    expect(c.colors.get("text2image/flux")).toBe("var(--series-3)");
  });
});

describe("token accounting hint", () => {
  it("fires for LLM calls without any token figures", () => {
    const m = { modality: "llm" as const, recipe: "x", jobs: 2, failed: 0, tasks: 2, last_ts: 1 };
    expect(predatesTokenAccounting(usage({ recipes: [{ ...m, tokens_in: null, tokens_out: null }] }))).toBe(true);
    expect(predatesTokenAccounting(usage({ recipes: [{ ...m, tokens_in: 5, tokens_out: null }] }))).toBe(false);
  });
});

const era = (over: Partial<GpuEra>): GpuEra => ({
  uuid: "GPU-a",
  name: "NVIDIA GeForce RTX 5090",
  total_gb: 32,
  first_seen: 100,
  last_seen: 200,
  jobs: 0,
  failed: 0,
  tokens_in: null,
  tokens_out: null,
  avg_run_ms: null,
  first_job: null,
  last_job: null,
  ...over,
});

describe("eras", () => {
  it("is current when the card is in the live GPU list", () => {
    expect(isCurrentEra(era({}), new Set(["GPU-a"]), 10_000)).toBe(true);
    expect(isCurrentEra(era({ last_seen: 9_999 }), new Set(["GPU-b"]), 10_000)).toBe(false);
  });
  it("falls back to recently seen without a live list", () => {
    expect(isCurrentEra(era({ last_seen: 9_900 }), null, 10_000)).toBe(true);
    expect(isCurrentEra(era({ last_seen: 9_000 }), null, 10_000)).toBe(false);
  });
  it("sorts newest first and sums tokens, null when never counted", () => {
    const list = erasNewestFirst([era({ uuid: "old", first_seen: 1 }), era({ uuid: "new", first_seen: 2 })]);
    expect(list.map((e) => e.uuid)).toEqual(["new", "old"]);
    expect(eraTokens(era({}))).toBeNull();
    expect(eraTokens(era({ tokens_in: 3, tokens_out: null }))).toBe(3);
  });
});

const appRow = (over: Partial<UsageAppRow>): UsageAppRow => ({
  app: "kw",
  modality: "llm",
  recipe: "qwen",
  jobs: 1,
  failed: 0,
  tokens_in: null,
  tokens_out: null,
  run_ms: null,
  last_ts: null,
  gpu_ms: null,
  energy_wh: null,
  energy_estimated: false,
  ...over,
});

describe("rollupApps", () => {
  it("sums an app's recipes, keeping null where nothing was counted", () => {
    const [kw] = rollupApps([
      appRow({ jobs: 2, failed: 1, tokens_in: 100, run_ms: 1000, gpu_ms: 800, energy_wh: 0.5, last_ts: 10 }),
      appRow({ modality: "text2image", recipe: "flux", jobs: 3, gpu_ms: 3000, energy_wh: 2, last_ts: 20 }),
    ]);
    expect(kw).toMatchObject({
      app: "kw",
      jobs: 5,
      failed: 1,
      tokens_in: 100,
      tokens_out: null,
      run_ms: 1000,
      gpu_ms: 3800,
      energy_wh: 2.5,
      last_ts: 20,
    });
    // Most calls first within the app.
    expect(kw!.recipes.map((r) => r.recipe)).toEqual(["flux", "qwen"]);
  });
  it("orders by energy, then GPU time, then calls, and keeps the unknown bucket apart", () => {
    const apps = rollupApps([
      appRow({ app: null, jobs: 50 }),
      appRow({ app: "anonymous", jobs: 4, gpu_ms: 100 }),
      appRow({ app: "kw", jobs: 1, energy_wh: 3, energy_estimated: true }),
      appRow({ app: "ha", jobs: 1, energy_wh: 1 }),
    ]);
    expect(apps.map((a) => a.app)).toEqual(["kw", "ha", "anonymous", null]);
    expect(apps[0]!.energy_estimated).toBe(true);
    expect(apps[1]!.energy_estimated).toBe(false);
    expect(appFilterValue(apps[3]!.app)).toBe(UNKNOWN);
  });
});

describe("appFromHash", () => {
  it("reads the app a link filters to", () => {
    expect(appFromHash("#/usage?app=kompost-wars")).toBe("kompost-wars");
    expect(appFromHash("#/usage?app=a%2Fb")).toBe("a/b");
    expect(appFromHash("#/usage")).toBe("");
  });
});

describe("appFilterOptions", () => {
  const entry = (name: string, revoked: string | null = null): AppEntry => ({
    name,
    description: null,
    created: "2026-11-20T10:00:00Z",
    revoked,
  });
  it("lists registered apps, then deleted ones the history has, then the two buckets", () => {
    const opts = appFilterOptions([entry("zz"), entry("aa", "2026-11-21T00:00:00Z")], ["old", "zz", null, ANONYMOUS], "");
    expect(opts.map((o) => o.value)).toEqual(["aa", "zz", "old", ANONYMOUS, UNKNOWN]);
    expect(opts[0]).toMatchObject({ revoked: true, registered: true });
    expect(opts[2]).toMatchObject({ registered: false });
    expect(opts.at(-1)!.kind).toBe("unknown");
  });
  it("keeps the current filter selectable before the app list arrives", () => {
    expect(appFilterOptions(undefined, [], "kw").map((o) => o.value)).toEqual(["kw", ANONYMOUS, UNKNOWN]);
    expect(appFilterOptions(undefined, [], UNKNOWN)).toHaveLength(2);
  });
});

describe("energySplit", () => {
  const energy = (over: Partial<UsageEnergy>): UsageEnergy => ({
    attributed_wh: 30,
    idle_wh: 90,
    estimated: false,
    cards: [{ gpu_uuid: "GPU-a", attributed_wh: 30, idle_wh: 90 }],
    ...over,
  });
  it("splits attributed from idle, with the share", () => {
    expect(energySplit(energy({}))).toMatchObject({ attributed: 30, idle: 90, total: 120, attributedPct: 25 });
  });
  it("lists cards only when there is more than one", () => {
    expect(energySplit(energy({}))!.cards).toEqual([]);
    const two = energySplit(
      energy({
        cards: [
          { gpu_uuid: "GPU-a", attributed_wh: 10, idle_wh: 40 },
          { gpu_uuid: "GPU-b", attributed_wh: 20, idle_wh: 50 },
        ],
      }),
    )!;
    expect(two.cards.map((c) => c.total)).toEqual([50, 70]);
  });
  it("counts loading and unloading on a line of its own, in the total", () => {
    const split = energySplit(energy({ load_wh: 80, cards: [] }))!;
    expect(split).toMatchObject({ attributed: 30, idle: 90, load: 80, total: 200, attributedPct: 15, loadPct: 40 });
  });
  it("reads a giq from before the load line as no load", () => {
    expect(energySplit(energy({}))).toMatchObject({ load: 0, loadPct: 0 });
  });
  it("has no idle line or share for one app, and none for an empty window", () => {
    expect(energySplit(energy({ idle_wh: null, cards: [] }))).toMatchObject({ idle: null, load: null, total: null, attributedPct: null });
    expect(energySplit(energy({ attributed_wh: 0, idle_wh: 0, cards: [] }))!.attributedPct).toBeNull();
    expect(energySplit(undefined)).toBeNull();
  });
  it("names a card from the eras", () => {
    expect(cardNameFor("GPU-a", [era({})])).toBe("NVIDIA GeForce RTX 5090");
    expect(cardNameFor("GPU-x", [era({})])).toBeNull();
    expect(cardNameFor(null, [era({})])).toBeNull();
  });
});
