// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import type { StatsUsage } from "../../api/types";
import { KpiTile } from "../../components";
import { useFormat } from "../../lib/useFormat";
import "./Usage.css";

/* The window's totals. Token figures are compact (1,2 Mio.), so the exact
   count rides in the tile's title; failed calls carry their share. GPU time
   and energy are what jobs were attributed (ADR-006 D5): the idle line is
   the energy card's, not part of these. */
export function UsageKpis({
  totals,
  estimated,
}: {
  totals: StatsUsage["totals"] | undefined;
  /** Some of the energy is from power samples rather than a card's counter. */
  estimated?: boolean;
}) {
  const { t } = useTranslation("usage");
  const f = useFormat();
  const exact = (n: number | undefined) =>
    n == null ? undefined : t("kpi.tokensExact", { count: n, n: f.num(n) });
  const failShare =
    totals && totals.jobs > 0 && totals.failed > 0
      ? t("kpi.failedShare", { pct: f.pct((totals.failed / totals.jobs) * 100, 1) })
      : undefined;
  return (
    <div className="us-kpis">
      <KpiTile label={t("kpi.calls")} value={f.num(totals?.jobs)} />
      <KpiTile label={t("kpi.tokensIn")} value={f.tok(totals?.tokens_in)} title={exact(totals?.tokens_in)} />
      <KpiTile label={t("kpi.tokensOut")} value={f.tok(totals?.tokens_out)} title={exact(totals?.tokens_out)} />
      <KpiTile
        label={t("kpi.failed")}
        value={f.num(totals?.failed)}
        className={totals?.failed ? "us-kpi-failed" : undefined}
      >
        {failShare}
      </KpiTile>
      <KpiTile label={t("kpi.gpuTime")} value={f.span(totals?.gpu_ms)} title={t("kpi.gpuTimeTitle")} />
      <KpiTile label={t("kpi.energy")} value={f.energy(totals?.energy_wh)} title={t("kpi.energyTitle")}>
        {estimated ? t("energy.estimate") : undefined}
      </KpiTile>
    </div>
  );
}
