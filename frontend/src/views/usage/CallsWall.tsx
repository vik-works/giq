// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import type { JobRecord } from "../../api/types";
import { Card, EmptyState, JobStatus, ModelLabel } from "../../components";
import { useFormat } from "../../lib/useFormat";
import { WALL_LIMIT } from "./useUsageData";
import "./Usage.css";

/* The last calls, newest first, in a scrolling box with a sticky header so
   the columns stay named as the reader scrolls. A failed call's error is
   server text: it shows as the row's native tooltip (a title attribute,
   escaped like any other text) rather than widening the table. */
export function CallsWall({ jobs, appFiltered }: { jobs: JobRecord[] | undefined; appFiltered?: boolean }) {
  const { t } = useTranslation("usage");
  const f = useFormat();
  // The call log is not kept per app, so under an app filter it says so.
  const meta = t(appFiltered ? "wall.metaAllApps" : "wall.meta", { count: WALL_LIMIT });
  return (
    <Card kicker={t("wall.kicker")} actions={meta}>
      {!jobs ? (
        <EmptyState compact>{t("common:empty.loading")}</EmptyState>
      ) : jobs.length === 0 ? (
        <EmptyState compact>{t("wall.empty")}</EmptyState>
      ) : (
        <div className="us-wall" tabIndex={0} role="region" aria-label={t("wall.kicker")}>
          <table className="table us-table">
            <thead>
              <tr>
                <th>{t("wall.when")}</th>
                <th>{t("wall.model")}</th>
                <th>{t("wall.status")}</th>
                <th className="num">{t("wall.queue")}</th>
                <th className="num">{t("wall.run")}</th>
                <th className="num">{t("wall.tokIn")}</th>
                <th className="num">{t("wall.tokOut")}</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                  <tr key={j.job_id} title={j.error ?? undefined}>
                    <td className="us-nowrap" title={f.dateTime(j.t)}>
                      {f.ago(j.t)}
                    </td>
                    <td className="us-nowrap">
                      <ModelLabel worker={j.modality} model={j.recipe} />
                    </td>
                    <td>
                      <JobStatus status={j.status} error={j.error} />
                    </td>
                    <td className="num">{f.dur(j.queue_ms)}</td>
                    <td className="num">{f.dur(j.run_ms)}</td>
                    <td className="num">{f.tok(j.tokens_in)}</td>
                    <td className="num">{f.tok(j.tokens_out)}</td>
                  </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
