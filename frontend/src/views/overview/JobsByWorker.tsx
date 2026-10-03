// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import type { StatsTimeline } from "../../api/types";
import { Card } from "../../components/Card";
import { WorkerIcon } from "../../components/WorkerIcon";
import { Legend, StackedBars, type BarBucket } from "../../components/charts";
import { workerColor, workerRank } from "../../lib/series";
import { useFormat } from "../../lib/useFormat";
import { useModalityLabel } from "../../lib/modalities";

type Point = StatsTimeline["points"][number];

/* Every bucket of the window is drawn, empty ones included, so a quiet night
   reads as a gap rather than being squeezed out. Workers stack in the fixed
   slot order, so a colour sits at the same height in every column. */
export function JobsByWorker({ timeline, hours }: { timeline: StatsTimeline | undefined; hours: number }) {
  const { t } = useTranslation("overview");
  const modalityLabel = useModalityLabel();
  const f = useFormat();
  const bucket = timeline?.bucket_s ?? 3600;
  const nowS = Date.now() / 1000;
  const start = Math.floor((nowS - hours * 3600) / bucket) * bucket;
  const n = Math.ceil((hours * 3600) / bucket);

  const byT = new Map<number, Map<string, Point>>();
  for (const p of timeline?.points ?? []) {
    if (!byT.has(p.t)) byT.set(p.t, new Map());
    byT.get(p.t)!.set(p.modality, p);
  }
  const workers = [...new Set((timeline?.points ?? []).map((p) => p.modality))].sort(
    (a, b) => workerRank(a) - workerRank(b),
  );
  const buckets: BarBucket[] = Array.from({ length: n }, (_, i) => {
    const bt = start + i * bucket;
    const per = byT.get(bt);
    return {
      key: bt,
      segments: workers.map((w) => ({ id: w, value: per?.get(w)?.jobs ?? 0, color: workerColor(w) })),
    };
  });
  const tooltip = (b: BarBucket) => {
    const per = byT.get(b.key as number);
    return (
      <>
        <b>{f.dateTime(b.key as number)}</b>
        {workers
          .filter((w) => per?.get(w))
          .map((w) => {
            const p = per!.get(w)!;
            return (
              <div key={w}>
                <WorkerIcon worker={w} size={12} /> {modalityLabel(w)}: {f.num(p.jobs)}
                {p.failed > 0 && <span className="text-critical"> ({t("common:chart.failed", { count: p.failed })})</span>}
              </div>
            );
          })}
      </>
    );
  };

  return (
    <Card title={t("usage.jobsByWorker")}>
      <StackedBars
        buckets={buckets}
        ariaLabel={t("usage.jobsByWorkerLabel")}
        tooltip={tooltip}
        xLabels={{ start: f.dayShort(start), end: t("common:chart.now") }}
        formatY={(v) => f.num(v)}
        emptyText={t("common:empty.noJobs")}
      />
      <Legend
        label={t("common:chart.legend")}
        items={workers.map((w) => ({
          id: w,
          color: workerColor(w),
          label: modalityLabel(w),
          icon: <WorkerIcon worker={w} size={13} />,
        }))}
      />
    </Card>
  );
}
