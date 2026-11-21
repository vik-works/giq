// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import type { GpuEra, UsagePeriod } from "../../api/types";
import { SegmentedControl, type SegmentOption } from "../../components";
import { shortGpuName } from "../../lib/cards";
import { useFormat } from "../../lib/useFormat";
import type { AppFilterOption } from "./usageModel";
import "./Usage.css";

export const PERIODS: readonly UsagePeriod[] = ["day", "week", "month", "all"];

export interface UsageFiltersProps {
  period: UsagePeriod;
  onPeriod: (p: UsagePeriod) => void;
  gpu: string;
  onGpu: (uuid: string) => void;
  /** Newest first, each with whether its card is installed now. */
  eras: { era: GpuEra; current: boolean }[];
  /** "" for every app; an app's name, "anonymous" or "unknown". */
  app: string;
  onApp: (app: string) => void;
  apps: AppFilterOption[];
}

/* The period, the card and the app the view is filtered to. Two eras can
   share a name (a card replaced by the same model), so a past era's option
   carries its dates; the installed card is marked current instead. */
export function UsageFilters({ period, onPeriod, gpu, onGpu, eras, app, onApp, apps }: UsageFiltersProps) {
  const { t } = useTranslation("usage");
  const f = useFormat();
  const options: SegmentOption<UsagePeriod>[] = PERIODS.map((p) => ({
    value: p,
    label: t(`period.${p}`),
  }));
  // A filter on a card that is no longer listed keeps its option, so the
  // select never shows a value it cannot display.
  const known = !gpu || eras.some((e) => e.era.uuid === gpu);
  return (
    <div className="us-filters">
      <SegmentedControl
        options={options}
        value={period}
        onChange={onPeriod}
        label={t("period.label")}
        size="sm"
      />
      <label className="us-gpu">
        <span className="us-gpu-label">{t("gpuFilter.label")}</span>
        <select className="input us-gpu-select" value={gpu} onChange={(e) => onGpu(e.target.value)}>
          <option value="">{t("gpuFilter.all")}</option>
          {!known && <option value={gpu}>{gpu}</option>}
          {eras.map(({ era, current }) => {
            const name = shortGpuName(era.name);
            return (
              <option key={era.uuid} value={era.uuid}>
                {current
                  ? t("gpuFilter.current", { name })
                  : t("gpuFilter.past", {
                      name,
                      from: f.dayShort(era.first_job ?? era.first_seen),
                      to: f.day(era.last_job ?? era.last_seen),
                    })}
              </option>
            );
          })}
        </select>
      </label>
      <label className="us-gpu">
        <span className="us-gpu-label">{t("appFilter.label")}</span>
        <select className="input us-gpu-select" value={app} onChange={(e) => onApp(e.target.value)}>
          <option value="">{t("appFilter.all")}</option>
          {apps.map((o) => (
            <option key={o.value} value={o.value}>
              {o.kind === "anonymous"
                ? t("app.anonymous")
                : o.kind === "unknown"
                  ? t("appFilter.unknown")
                  : o.revoked
                    ? t("appFilter.revoked", { name: o.value })
                    : !o.registered
                      ? t("appFilter.deleted", { name: o.value })
                      : o.value}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
