// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* Every number, date and unit on screen goes through here, formatted by Intl
   for the active language: "1,234.5 GB" in English is "1.234,5 GB" in German,
   and a hand-rolled toFixed() gets that wrong in one of them. Each function
   takes the language explicitly so it stays pure (and testable); components
   use the bound set from useFormat(). A null/undefined value renders as an
   en dash, the one placeholder the whole dashboard uses for "no reading". */

export const DASH = "–";

type Num = number | null | undefined;

// Intl constructors are not free, and charts format hundreds of ticks.
const cache = new Map<string, Intl.NumberFormat>();
function nf(lang: string, opts: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = lang + JSON.stringify(opts);
  let f = cache.get(key);
  if (!f) {
    f = new Intl.NumberFormat(lang, opts);
    cache.set(key, f);
  }
  return f;
}

const fixed = (digits: number): Intl.NumberFormatOptions => ({
  minimumFractionDigits: digits,
  maximumFractionDigits: digits,
});

/* German leaves four-digit numbers ungrouped by default ("2118" but
   "56.904"), which makes a column of counts look like two formats. Tables
   and tiles read better with every figure grouped the same way. */
const GROUP: Intl.NumberFormatOptions = { useGrouping: "always" };

/** A plain number, grouped for the language. */
export function fmtNum(n: Num, lang: string, digits = 0): string {
  return n == null ? DASH : nf(lang, { ...GROUP, ...fixed(digits) }).format(n);
}

/** Bytes in the largest unit that keeps the figure readable (decimal units, as disks are sold). */
export function fmtBytes(b: Num, lang: string): string {
  if (b == null) return DASH;
  const unit = (u: string, v: number, d: number) =>
    nf(lang, { style: "unit", unit: u, ...fixed(d) }).format(v);
  if (b >= 1e12) return unit("terabyte", b / 1e12, 2);
  if (b >= 1e9) return unit("gigabyte", b / 1e9, 1);
  if (b >= 1e6) return unit("megabyte", b / 1e6, 0);
  return unit("kilobyte", Math.round(b / 1e3), 0);
}

/** A figure already in GB (VRAM), one decimal by default. */
export function fmtGB(gb: Num, lang: string, digits = 1): string {
  return gb == null ? DASH : nf(lang, { style: "unit", unit: "gigabyte", ...fixed(digits) }).format(gb);
}

/** A duration in milliseconds: ms under a second, seconds under 90 s, minutes beyond. */
export function fmtDur(ms: Num, lang: string): string {
  if (ms == null) return DASH;
  const unit = (u: string, v: number, d: number) =>
    nf(lang, { style: "unit", unit: u, unitDisplay: "short", maximumFractionDigits: d }).format(v);
  if (ms < 1000) return unit("millisecond", Math.round(ms), 0);
  if (ms < 90_000) return unit("second", ms / 1000, 1);
  return unit("minute", ms / 60_000, 1);
}

/* A long total (GPU time over a month): fmtDur's minutes stop being readable
   past an hour and a half, so hours take over there. */
/** A total duration in milliseconds: like fmtDur, and hours from 90 minutes on. */
export function fmtSpan(ms: Num, lang: string): string {
  if (ms == null || ms < 5_400_000) return fmtDur(ms, lang);
  const d = ms < 36_000_000 ? 1 : 0;
  return nf(lang, { style: "unit", unit: "hour", unitDisplay: "short", ...GROUP, maximumFractionDigits: d }).format(
    ms / 3_600_000,
  );
}

/* Energy per job is fractions of a watt-hour, a month of a rig is kilowatt-
   hours. Intl has no watt-hour unit, so the symbol is appended as for watts;
   the precision shrinks as the figure grows, so a column stays narrow. */
/** Watt-hours: Wh below 1000, kWh from there. */
export function fmtEnergy(wh: Num, lang: string): string {
  if (wh == null) return DASH;
  if (wh >= 1000) {
    const kwh = wh / 1000;
    return `${nf(lang, { ...GROUP, ...fixed(kwh < 100 ? 2 : 1) }).format(kwh)} kWh`;
  }
  const digits = wh === 0 ? 0 : wh < 1 ? 2 : wh < 10 ? 1 : 0;
  return `${nf(lang, { ...GROUP, ...fixed(digits) }).format(wh)} Wh`;
}

/** "5 s ago" / "vor 5 s" for an epoch-seconds timestamp. */
export function fmtAgo(t: Num, lang: string, nowMs: number = Date.now()): string {
  if (t == null) return DASH;
  const rtf = new Intl.RelativeTimeFormat(lang, { style: "narrow", numeric: "always" });
  const s = Math.max(0, nowMs / 1000 - t);
  if (s < 90) return rtf.format(-Math.round(s), "second");
  if (s < 5400) return rtf.format(-Math.round(s / 60), "minute");
  if (s < 172_800) return rtf.format(-Math.round(s / 3600), "hour");
  return rtf.format(-Math.round(s / 86_400), "day");
}

/** Token counts: compact for the language (12.3K / 12.345, 1.3M / 1,3 Mio.). */
export function fmtTok(n: Num, lang: string): string {
  if (n == null) return DASH;
  return nf(lang, { ...GROUP, notation: "compact", maximumFractionDigits: 1 }).format(n);
}

/** A 0–100 percentage (GPU utilisation, fan) as the language writes it. */
export function fmtPct(v: Num, lang: string, digits = 0): string {
  return v == null ? DASH : nf(lang, { style: "percent", ...fixed(digits) }).format(v / 100);
}

/** Watts. Intl has no watt unit, so the symbol is appended after a narrow no-break space. */
export function fmtWatts(w: Num, lang: string): string {
  return w == null ? DASH : `${nf(lang, fixed(0)).format(w)} W`;
}

/** Degrees Celsius, whole degrees. */
export function fmtTemp(c: Num, lang: string): string {
  return c == null ? DASH : nf(lang, { style: "unit", unit: "celsius", ...fixed(0) }).format(c);
}

/** Tokens per second, one decimal under ten, whole above. */
export function fmtRate(v: Num, lang: string): string {
  if (v == null) return DASH;
  return nf(lang, fixed(v < 10 ? 1 : 0)).format(v);
}

const toDate = (t: number) => new Date(t * 1000);

/** Clock time "14:05" for an epoch-seconds timestamp. */
export function fmtTime(t: Num, lang: string, seconds = false): string {
  if (t == null) return DASH;
  return toDate(t).toLocaleTimeString(lang, {
    hour: "2-digit",
    minute: "2-digit",
    ...(seconds ? { second: "2-digit" } : {}),
  });
}

/** "Sep 28, 14:05" — a bucket or sample label with its day. */
export function fmtDateTime(t: Num, lang: string): string {
  if (t == null) return DASH;
  return toDate(t).toLocaleString(lang, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** "Sep 28, 2026" — a calendar day. */
export function fmtDay(t: Num, lang: string): string {
  if (t == null) return DASH;
  return toDate(t).toLocaleDateString(lang, { year: "numeric", month: "short", day: "numeric" });
}

/** "Sep 28" — a day without its year, for axis ends. */
export function fmtDayShort(t: Num, lang: string): string {
  if (t == null) return DASH;
  return toDate(t).toLocaleDateString(lang, { month: "short", day: "numeric" });
}

/** Uptime from seconds: the two largest units ("6 days, 4 hr" / "6 Tg., 4 Std."). */
export function fmtUptime(s: Num, lang: string): string {
  if (s == null) return DASH;
  const total = Math.max(0, Math.floor(s));
  // DurationFormat drops zero-valued units, which would leave nothing to show.
  if (total < 60) {
    return nf(lang, { style: "unit", unit: "second", unitDisplay: "short" }).format(total);
  }
  const parts = {
    days: Math.floor(total / 86_400),
    hours: Math.floor((total % 86_400) / 3600),
    minutes: Math.floor((total % 3600) / 60),
    seconds: total % 60,
  };
  const order = ["days", "hours", "minutes", "seconds"] as const;
  const first = order.findIndex((k) => parts[k] > 0);
  const keep = order.slice(first, first + 2);
  const shown = Object.fromEntries(keep.map((k) => [k, parts[k as keyof typeof parts]]));
  type DurationFormatCtor = new (
    l: string,
    o: { style: string },
  ) => { format(d: Record<string, number>): string };
  const DF = (Intl as unknown as { DurationFormat?: DurationFormatCtor }).DurationFormat;
  if (DF) return new DF(lang, { style: "short" }).format(shown);
  // Browsers before Intl.DurationFormat: unit-formatted pieces, joined.
  const unit = { days: "day", hours: "hour", minutes: "minute", seconds: "second" } as const;
  return keep
    .map((k) =>
      nf(lang, { style: "unit", unit: unit[k as keyof typeof unit], unitDisplay: "short" }).format(
        shown[k] ?? 0,
      ),
    )
    .join(" ");
}

/** Truncate with an ellipsis for fixed-width chart labels (full text goes in the tooltip). */
export function ellipsize(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}
