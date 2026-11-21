// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import * as f from "./format";

/** The formatters of ./format bound to the active language; re-bound when it changes. */
export function useFormat() {
  const { i18n } = useTranslation();
  const lang = i18n.resolvedLanguage ?? i18n.language ?? "en";
  return useMemo(
    () => ({
      lang,
      num: (n: number | null | undefined, digits?: number) => f.fmtNum(n, lang, digits),
      bytes: (b: number | null | undefined) => f.fmtBytes(b, lang),
      gb: (gb: number | null | undefined, digits?: number) => f.fmtGB(gb, lang, digits),
      dur: (ms: number | null | undefined) => f.fmtDur(ms, lang),
      span: (ms: number | null | undefined) => f.fmtSpan(ms, lang),
      energy: (wh: number | null | undefined) => f.fmtEnergy(wh, lang),
      ago: (t: number | null | undefined) => f.fmtAgo(t, lang),
      tok: (n: number | null | undefined) => f.fmtTok(n, lang),
      pct: (v: number | null | undefined, digits?: number) => f.fmtPct(v, lang, digits),
      watts: (w: number | null | undefined) => f.fmtWatts(w, lang),
      temp: (c: number | null | undefined) => f.fmtTemp(c, lang),
      rate: (v: number | null | undefined) => f.fmtRate(v, lang),
      time: (t: number | null | undefined, seconds?: boolean) => f.fmtTime(t, lang, seconds),
      dateTime: (t: number | null | undefined) => f.fmtDateTime(t, lang),
      day: (t: number | null | undefined) => f.fmtDay(t, lang),
      dayShort: (t: number | null | undefined) => f.fmtDayShort(t, lang),
      uptime: (s: number | null | undefined) => f.fmtUptime(s, lang),
    }),
    [lang],
  );
}

export type Formatters = ReturnType<typeof useFormat>;
