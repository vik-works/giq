// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import type { GpuEra, UsageEnergy } from "../../api/types";
import { Card, EmptyState } from "../../components";
import { shortGpuName } from "../../lib/cards";
import { useFormat } from "../../lib/useFormat";
import { cardNameFor, energySplit } from "./usageModel";
import "./EnergyCard.css";

/* What the cards drew in the window, in two parts (ADR-006 D5): what jobs
   drew, split among the jobs that ran at the same time, and two lines that
   belong to nobody: idle and residency, and loading and unloading
   instances. Filtered to one app there are neither, and the card says why
   instead of showing a zero. One bar for
   the split; a table per card only when there is more than one card, as
   with one it would repeat the bar. */
export function EnergyCard({ energy, eras }: { energy: UsageEnergy | undefined; eras: GpuEra[] | undefined }) {
  const { t } = useTranslation("usage");
  const f = useFormat();
  const split = energySplit(energy);
  const name = (uuid: string | null) => {
    const n = cardNameFor(uuid, eras);
    return n ? shortGpuName(n) : (uuid ?? t("energy.unknownCard"));
  };
  return (
    <Card
      kicker={t("energy.kicker")}
      actions={
        split?.estimated ? (
          <span className="us-en-estimate" title={t("energy.estimateTitle")}>
            {t("energy.estimate")}
          </span>
        ) : undefined
      }
    >
      {!split ? (
        <EmptyState compact>{t("common:empty.loading")}</EmptyState>
      ) : split.total == null ? (
        <div className="us-en-body">
          <div className="us-en-figure">
            <span className="us-en-value">{f.energy(split.attributed)}</span>
            <span className="us-en-label">{t("energy.attributed")}</span>
          </div>
          <p className="hint">{t("energy.appOnly")}</p>
        </div>
      ) : split.total === 0 ? (
        <EmptyState compact>{t("energy.empty")}</EmptyState>
      ) : (
        <div className="us-en-body">
          <div
            className="us-en-bar"
            role="img"
            aria-label={t("energy.barAria", {
              attributed: f.energy(split.attributed),
              idle: f.energy(split.idle),
              load: f.energy(split.load),
            })}
          >
            <span className="us-en-bar-jobs" style={{ width: `${split.attributedPct ?? 0}%` }} />
            <span className="us-en-bar-load" style={{ width: `${split.loadPct ?? 0}%` }} />
          </div>
          <div className="us-en-legend">
            <div className="us-en-figure">
              <span className="us-en-swatch us-en-swatch-jobs" aria-hidden />
              <span className="us-en-value">{f.energy(split.attributed)}</span>
              <span className="us-en-label">
                {t("energy.attributed")} · {f.pct(split.attributedPct)}
              </span>
            </div>
            <div className="us-en-figure">
              <span className="us-en-swatch us-en-swatch-idle" aria-hidden />
              <span className="us-en-value">{f.energy(split.idle)}</span>
              <span className="us-en-label">{t("energy.idle")}</span>
            </div>
            <div className="us-en-figure" title={t("energy.loadTitle")}>
              <span className="us-en-swatch us-en-swatch-load" aria-hidden />
              <span className="us-en-value">{f.energy(split.load)}</span>
              <span className="us-en-label">{t("energy.load")}</span>
            </div>
            <div className="us-en-figure">
              <span className="us-en-value">{f.energy(split.total)}</span>
              <span className="us-en-label">{t("energy.total")}</span>
            </div>
          </div>
          {split.cards.length > 0 && (
            <div className="table-wrap">
              <table className="table us-table">
                <thead>
                  <tr>
                    <th>{t("energy.card")}</th>
                    <th className="num">{t("energy.attributed")}</th>
                    <th className="num">{t("energy.idleShort")}</th>
                    <th className="num">{t("energy.loadShort")}</th>
                    <th className="num">{t("energy.totalShort")}</th>
                  </tr>
                </thead>
                <tbody>
                  {split.cards.map((c) => (
                    <tr key={c.gpu_uuid ?? ""}>
                      <td title={c.gpu_uuid ?? undefined}>{name(c.gpu_uuid)}</td>
                      <td className="num">{f.energy(c.attributed)}</td>
                      <td className="num">{f.energy(c.idle)}</td>
                      <td className="num">{f.energy(c.load)}</td>
                      <td className="num">{f.energy(c.total)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="hint">{t("energy.rule")}</p>
        </div>
      )}
    </Card>
  );
}
