// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { AppEntry, GpuEra, StatsUsage, UsageAppRow, UsageEnergy, UsagePeriod } from "../../api/types";
import type { BarBucket } from "../../components/charts";
import { assignSeriesColors } from "../../lib/series";

/* The usage view's arithmetic, kept free of React so it can be tested
   without a DOM: the chart's x-domain, its bucket labels, the stacks,
   which card era is the current one, and the per-app rollup and energy
   split (ADR-006). */

type Row = { modality: string; recipe: string };

/* Keyed on modality and recipe, not the recipe alone: flux_klein renders and
   edits from one recipe, and a render and an edit are different work. */
export const modelKey = (m: Row): string => `${m.modality}/${m.recipe}`;

const tokens = (r: { tokens_in: number | null; tokens_out: number | null }): number =>
  (r.tokens_in ?? 0) + (r.tokens_out ?? 0);

const pad2 = (n: number) => String(n).padStart(2, "0");
/** The server's local-time day key, "YYYY-MM-DD". */
export const dayKey = (d: Date): string =>
  `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;

/* The full x-domain of bucket keys, so quiet hours and days render as gaps
   rather than as missing columns that squeeze the rest together. The keys
   match what /stats/usage emits (strftime in the server's local time):
   "YYYY-MM-DD HH:00" for a day, "YYYY-MM-DD" for a week or a month,
   "YYYY-MM" for all time — where the domain is only the months actually
   seen, since "all" has no fixed start. */
export function usageDomain(period: UsagePeriod, present: string[], now = new Date()): string[] {
  if (period === "all") return [...new Set(present)].sort();
  const keys: string[] = [];
  if (period === "day") {
    for (let h = 0; h <= now.getHours(); h++) keys.push(`${dayKey(now)} ${pad2(h)}:00`);
    return keys;
  }
  const days = period === "week" ? 7 : 30;
  for (let i = days - 1; i >= 0; i--) {
    // Calendar arithmetic, not now − i·24 h, so a DST switch cannot skip or repeat a day.
    keys.push(dayKey(new Date(now.getFullYear(), now.getMonth(), now.getDate() - i)));
  }
  return keys;
}

/** A bucket key back to the local Date it starts at. */
export function bucketDate(b: string): Date {
  const [date = "", time] = b.split(" ");
  const [y = 1970, m = 1, d = 1] = date.split("-").map(Number);
  const h = time ? Number(time.slice(0, 2)) : 0;
  return new Date(y, m - 1, d, h);
}

const dtf = new Map<string, Intl.DateTimeFormat>();
function df(lang: string, opts: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  const key = lang + JSON.stringify(opts);
  let f = dtf.get(key);
  if (!f) dtf.set(key, (f = new Intl.DateTimeFormat(lang, opts)));
  return f;
}

/** Axis label for a bucket: the hour for a day, the date for a week or month, month and year for all time. */
export function bucketLabel(b: string, period: UsagePeriod, lang: string): string {
  const d = bucketDate(b);
  if (period === "day") return df(lang, { hour: "2-digit", minute: "2-digit" }).format(d);
  if (period === "all") return df(lang, { month: "short", year: "numeric" }).format(d);
  return df(lang, { month: "short", day: "numeric" }).format(d);
}

/** Tooltip heading for a bucket: a little more than the axis has room for. */
export function bucketTitle(b: string, period: UsagePeriod, lang: string): string {
  const d = bucketDate(b);
  if (period === "day") {
    const end = new Date(d.getTime() + 3600_000);
    return df(lang, { hour: "2-digit", minute: "2-digit" }).formatRange(d, end);
  }
  if (period === "all") return df(lang, { month: "long", year: "numeric" }).format(d);
  return df(lang, { weekday: "short", month: "short", day: "numeric" }).format(d);
}

export type SeriesPoint = StatsUsage["series"][number];

export interface UsageChart {
  /** Every bucket of the domain; each segment id is a model key, value its tokens. */
  buckets: BarBucket[];
  /** The domain keys, parallel to buckets (for labels and tooltips). */
  keys: string[];
  /** Per bucket: the series points behind its segments, bottom to top. */
  points: SeriesPoint[][];
  /** The models that have tokens in the window, in stack/legend order. */
  models: Row[];
  colors: Map<string, string>;
}

/* Stacks follow the per-model table's order (most tokens first, bottom of
   the stack), and only models with tokens take part: an image model's calls
   carry no tokens and would be an empty legend entry. Colours are assigned
   over every model in the window, alphabetically, so a model keeps its
   colour when a period or GPU switch changes what else is on screen. */
export function buildUsageChart(d: StatsUsage, period: UsagePeriod, now = new Date()): UsageChart {
  const colors = assignSeriesColors(d.recipes.map(modelKey));
  const models = d.recipes.filter((m) => tokens(m) > 0);
  const order = models.map(modelKey);
  const perBucket = new Map<string, Map<string, SeriesPoint>>();
  for (const p of d.series) {
    let per = perBucket.get(p.b);
    if (!per) perBucket.set(p.b, (per = new Map()));
    per.set(modelKey(p), p);
  }
  const keys = usageDomain(
    period,
    d.series.map((p) => p.b),
    now,
  );
  const points: SeriesPoint[][] = [];
  const buckets = keys.map((key) => {
    const per = perBucket.get(key);
    const stack = order.flatMap((k) => {
      const p = per?.get(k);
      return p && tokens(p) > 0 ? [p] : [];
    });
    points.push(stack);
    return {
      key,
      segments: stack.map((p) => ({
        id: modelKey(p),
        value: tokens(p),
        color: colors.get(modelKey(p)) ?? "var(--series-other)",
      })),
    };
  });
  return { buckets, keys, points, models, colors };
}

/* Token accounting starts later than the call log: an LLM call recorded
   before it counts as a call with no tokens. When such calls fall in the
   window the chart undercounts, and the hint says so. */
export const predatesTokenAccounting = (d: StatsUsage): boolean =>
  d.recipes.some(
    (m) => m.modality === "llm" && m.jobs > 0 && m.tokens_in == null && m.tokens_out == null,
  );

/** Eras seen within this long are "current" when the live GPU list is unavailable. */
const CURRENT_WINDOW_S = 300;

/* An era is current when its card is in the machine now. The live /gpus
   list says so directly; without it (nvidia-smi failing, first paint),
   an era whose card was sampled in the last five minutes counts, which is
   what the sampler's cadence guarantees for an installed card. */
export function isCurrentEra(e: GpuEra, liveUuids: Set<string> | null, nowS = Date.now() / 1000): boolean {
  if (liveUuids && liveUuids.size) return liveUuids.has(e.uuid);
  return e.last_seen > nowS - CURRENT_WINDOW_S;
}

/** Newest first: the current card is the one the reader cares about. */
export const erasNewestFirst = (eras: GpuEra[]): GpuEra[] =>
  [...eras].sort((a, b) => b.first_seen - a.first_seen);

/** Tokens of an era, or null when none were recorded (it predates accounting). */
export const eraTokens = (e: GpuEra): number | null =>
  e.tokens_in == null && e.tokens_out == null ? null : tokens(e);

// --- apps (ADR-006) ----------------------------------------------------------

/** The bucket for callers without a valid key; a name like any app's. */
export const ANONYMOUS = "anonymous";
/** The filter value for jobs recorded before apps existed (their app is null). */
export const UNKNOWN = "unknown";

/** The `?app=` value that selects a row's app: null (before apps) is "unknown". */
export const appFilterValue = (app: string | null): string => app ?? UNKNOWN;

/** The app a link asked for (#/usage?app=…), or "" for every app. */
export function appFromHash(hash: string): string {
  const query = hash.split("?")[1];
  return (query ? new URLSearchParams(query).get("app") : null) ?? "";
}

/* A sum that stays null when every part is: a recipe without token
   accounting, or jobs from before GPU time was recorded, show a dash, not a
   zero that claims nothing was used. */
const addNullable = (a: number | null, b: number | null): number | null =>
  a == null && b == null ? null : (a ?? 0) + (b ?? 0);

export interface AppRollup {
  /** The app as the server names it; null for jobs from before apps. */
  app: string | null;
  jobs: number;
  failed: number;
  tokens_in: number | null;
  tokens_out: number | null;
  run_ms: number | null;
  gpu_ms: number | null;
  energy_wh: number | null;
  energy_estimated: boolean;
  last_ts: number | null;
  /** The app's rows, one per modality and recipe, most calls first. */
  recipes: UsageAppRow[];
}

/* One row per app over its recipes. Ordered by what it cost the rig:
   energy, then GPU time, then calls, so the app worth asking about is on
   top; the name breaks ties so the order holds still between polls. */
export function rollupApps(rows: UsageAppRow[]): AppRollup[] {
  const by = new Map<string, AppRollup>();
  for (const r of rows) {
    const k = appFilterValue(r.app);
    let a = by.get(k);
    if (!a) {
      a = {
        app: r.app,
        jobs: 0,
        failed: 0,
        tokens_in: null,
        tokens_out: null,
        run_ms: null,
        gpu_ms: null,
        energy_wh: null,
        energy_estimated: false,
        last_ts: null,
        recipes: [],
      };
      by.set(k, a);
    }
    a.jobs += r.jobs;
    a.failed += r.failed;
    a.tokens_in = addNullable(a.tokens_in, r.tokens_in);
    a.tokens_out = addNullable(a.tokens_out, r.tokens_out);
    a.run_ms = addNullable(a.run_ms, r.run_ms);
    a.gpu_ms = addNullable(a.gpu_ms, r.gpu_ms);
    a.energy_wh = addNullable(a.energy_wh, r.energy_wh);
    a.energy_estimated ||= r.energy_estimated;
    a.last_ts = r.last_ts == null ? a.last_ts : Math.max(a.last_ts ?? 0, r.last_ts);
    a.recipes.push(r);
  }
  for (const a of by.values()) a.recipes.sort((x, y) => y.jobs - x.jobs);
  return [...by.values()].sort(
    (x, y) =>
      (y.energy_wh ?? 0) - (x.energy_wh ?? 0) ||
      (y.gpu_ms ?? 0) - (x.gpu_ms ?? 0) ||
      y.jobs - x.jobs ||
      appFilterValue(x.app).localeCompare(appFilterValue(y.app)),
  );
}

export interface AppFilterOption {
  value: string;
  kind: "app" | "anonymous" | "unknown";
  /** A registered app whose key no longer works. */
  revoked: boolean;
  /** False for a name only the history knows: an app deleted since. */
  registered: boolean;
}

/* The app filter's choices: the registered apps by name, then names the
   window's history carries that are no longer registered (deleted apps
   keep their jobs), then the two buckets that are not apps. The current
   value is always among them, so the select never holds a value it cannot
   show. */
export function appFilterOptions(
  registered: AppEntry[] | undefined,
  seen: (string | null)[],
  current: string,
): AppFilterOption[] {
  const reserved = new Set([ANONYMOUS, UNKNOWN]);
  const named = new Map<string, AppFilterOption>();
  for (const a of registered ?? []) {
    named.set(a.name, { value: a.name, kind: "app", revoked: !!a.revoked, registered: true });
  }
  for (const name of [...seen, current]) {
    if (!name || reserved.has(name) || named.has(name)) continue;
    named.set(name, { value: name, kind: "app", revoked: false, registered: false });
  }
  const apps = [...named.values()].sort(
    (a, b) => Number(b.registered) - Number(a.registered) || a.value.localeCompare(b.value),
  );
  return [
    ...apps,
    { value: ANONYMOUS, kind: "anonymous", revoked: false, registered: false },
    { value: UNKNOWN, kind: "unknown", revoked: false, registered: false },
  ];
}

export interface CardEnergy {
  gpu_uuid: string | null;
  attributed: number;
  idle: number;
  total: number;
}

export interface EnergySplit {
  attributed: number;
  /** Null when filtered to one app: idle energy belongs to nobody. */
  idle: number | null;
  /** Attributed plus idle; null with idle. */
  total: number | null;
  /** The attributed share of the total, 0–100; null without a total or with a zero one. */
  attributedPct: number | null;
  estimated: boolean;
  /** Per card, only when there is more than one: with one card it repeats the totals. */
  cards: CardEnergy[];
}

/** What the cards drew in the window: what jobs did, and the idle/residency line. */
export function energySplit(e: UsageEnergy | undefined): EnergySplit | null {
  if (!e) return null;
  const total = e.idle_wh == null ? null : e.attributed_wh + e.idle_wh;
  const cards =
    e.cards.length > 1
      ? e.cards.map((c) => ({
          gpu_uuid: c.gpu_uuid,
          attributed: c.attributed_wh,
          idle: c.idle_wh,
          total: c.attributed_wh + c.idle_wh,
        }))
      : [];
  return {
    attributed: e.attributed_wh,
    idle: e.idle_wh,
    total,
    attributedPct: total ? (e.attributed_wh / total) * 100 : null,
    estimated: e.estimated,
    cards,
  };
}

/** The card a uuid names, from the era list, or null when no era knows it. */
export function cardNameFor(uuid: string | null, eras: GpuEra[] | undefined): string | null {
  if (!uuid) return null;
  return eras?.find((e) => e.uuid === uuid)?.name ?? null;
}
