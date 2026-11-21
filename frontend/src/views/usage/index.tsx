// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { UsagePeriod } from "../../api/types";
import { errorText } from "../../api/client";
import { PageHeader } from "../../components";
import { useApps, useGpus } from "../../state";
import { AppsTable } from "./AppsTable";
import { CallsWall } from "./CallsWall";
import { EnergyCard } from "./EnergyCard";
import { ErasTable } from "./ErasTable";
import { ModelsTable } from "./ModelsTable";
import { TokensChart } from "./TokensChart";
import { UsageFilters } from "./UsageFilters";
import { UsageKpis } from "./UsageKpis";
import { appFilterOptions, appFromHash, erasNewestFirst, isCurrentEra } from "./usageModel";
import { useUsageData } from "./useUsageData";
import "./Usage.css";

/* Usage: what ran, how much it cost in tokens, GPU time and energy, for
   which app and on which card. Period, GPU and app filter the totals, the
   chart and the per-recipe and per-app tables; the GPU also filters the
   call wall (which is always the last hundred, whatever the period); the
   card table is the machine's whole history either way. An app filter
   arrives from the Apps page as #/usage?app=<name>. */
export default function UsageView() {
  const { t } = useTranslation("usage");
  const [period, setPeriod] = useState<UsagePeriod>("week");
  const [gpu, setGpu] = useState("");
  const [app, setApp] = useState(() => appFromHash(window.location.hash));
  const { eras, usage, jobs, stale } = useUsageData(period, gpu, app);
  const gpus = useGpus();
  const registered = useApps().data?.apps;

  const liveUuids = useMemo(
    () => (gpus.data ? new Set(gpus.data.gpus.map((g) => g.uuid)) : null),
    [gpus.data],
  );
  const eraRows = useMemo(
    () =>
      eras.data
        ? erasNewestFirst(eras.data.eras).map((era) => ({ era, current: isCurrentEra(era, liveUuids) }))
        : undefined,
    [eras.data, liveUuids],
  );
  const appOptions = useMemo(
    () => appFilterOptions(registered, (usage.data?.apps ?? []).map((r) => r.app), app),
    [registered, usage.data?.apps, app],
  );

  // Until the refetch for a new filter lands, the figures on screen belong
  // to the old one: dim them rather than present them as the answer.
  const error = usage.error ?? jobs.error ?? eras.error;

  return (
    <>
      <PageHeader
        title={t("title")}
        actions={
          <UsageFilters
            period={period}
            onPeriod={setPeriod}
            gpu={gpu}
            onGpu={setGpu}
            eras={eraRows ?? []}
            app={app}
            onApp={setApp}
            apps={appOptions}
          />
        }
      />
      {error ? (
        <p className="error-box" role="alert">
          {t("common:empty.failed", { error: errorText(error) })}
        </p>
      ) : null}
      <div className={`us-view stack${stale ? " us-stale" : ""}`} aria-busy={stale}>
        <UsageKpis totals={usage.data?.totals} estimated={usage.data?.energy.estimated} />
        <TokensChart data={usage.data} period={period} />
        <div className="us-split">
          <AppsTable rows={usage.data?.apps} />
          <EnergyCard energy={usage.data?.energy} eras={eras.data?.eras} />
        </div>
        <ErasTable eras={eraRows} />
        <ModelsTable data={usage.data} />
        <CallsWall jobs={jobs.data} appFiltered={!!app} />
      </div>
    </>
  );
}
