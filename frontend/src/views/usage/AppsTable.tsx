// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { CaretDownIcon, CaretRightIcon } from "@phosphor-icons/react";
import { Fragment, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { UsageAppRow } from "../../api/types";
import { Card, EmptyState, FailedCount, Icon, ModelLabel } from "../../components";
import { hrefFor } from "../../lib/useHashRoute";
import { useFormat, type Formatters } from "../../lib/useFormat";
import { appLabel } from "./appLabel";
import { appFilterValue, rollupApps, UNKNOWN } from "./usageModel";
import "./AppsTable.css";

type Figures = Pick<
  UsageAppRow,
  "jobs" | "failed" | "tokens_in" | "tokens_out" | "run_ms" | "gpu_ms" | "energy_wh" | "energy_estimated" | "last_ts"
>;

const tokens = (r: Figures) =>
  r.tokens_in == null && r.tokens_out == null ? null : (r.tokens_in ?? 0) + (r.tokens_out ?? 0);

/** The figure cells an app row and its recipe rows share. */
function FigureCells({ r, f, estimate }: { r: Figures; f: Formatters; estimate: string }) {
  return (
    <>
      <td className="num">{f.num(r.jobs)}</td>
      <td className="num">
        <FailedCount n={r.failed} />
      </td>
      <td className="num">{f.tok(tokens(r))}</td>
      <td className="num">{f.span(r.run_ms)}</td>
      <td className="num">{f.span(r.gpu_ms)}</td>
      <td className="num" title={r.energy_estimated ? estimate : undefined}>
        {f.energy(r.energy_wh)}
        {r.energy_estimated && r.energy_wh != null ? <span className="us-app-est">*</span> : null}
      </td>
      <td className="us-gap" title={r.last_ts ? f.dateTime(r.last_ts) : undefined}>
        {f.ago(r.last_ts)}
      </td>
    </>
  );
}

/* The window per app (ADR-006), costliest first. A row opens onto its
   recipes; that is all the breakdown there is, so the table stays one
   level deep. Jobs from before apps sit in their own "unknown" row, never
   folded into "anonymous", and an asterisk marks energy that is partly
   estimated. */
export function AppsTable({ rows }: { rows: UsageAppRow[] | undefined }) {
  const { t } = useTranslation("usage");
  const f = useFormat();
  const apps = useMemo(() => (rows ? rollupApps(rows) : undefined), [rows]);
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const toggle = (k: string) =>
    setOpen((s) => {
      const next = new Set(s);
      if (!next.delete(k)) next.add(k);
      return next;
    });
  const estimate = t("energy.estimateTitle");
  return (
    <Card
      kicker={t("apps.kicker")}
      actions={
        <a className="btn btn-ghost btn-sm" href={hrefFor("apps")}>
          {t("apps.manage")}
        </a>
      }
    >
      {!apps ? (
        <EmptyState compact>{t("common:empty.loading")}</EmptyState>
      ) : apps.length === 0 ? (
        <EmptyState compact>{t("apps.empty")}</EmptyState>
      ) : (
        <div className="table-wrap">
          <table className="table us-table">
            <thead>
              <tr>
                <th>{t("apps.app")}</th>
                <th className="num">{t("apps.calls")}</th>
                <th className="num">{t("apps.failed")}</th>
                <th className="num">{t("apps.tokens")}</th>
                <th className="num">{t("apps.run")}</th>
                <th className="num">{t("apps.gpu")}</th>
                <th className="num">{t("apps.energy")}</th>
                <th className="us-gap">{t("apps.lastUsed")}</th>
              </tr>
            </thead>
            <tbody>
              {apps.map((a) => {
                const k = appFilterValue(a.app);
                const expanded = open.has(k);
                const label = appLabel(a.app, t);
                return (
                  <Fragment key={k}>
                    <tr className="us-app-row">
                      <td>
                        <button
                          type="button"
                          className="us-app-toggle"
                          aria-expanded={expanded}
                          onClick={() => toggle(k)}
                          title={k === UNKNOWN ? t("app.unknownTitle") : undefined}
                        >
                          <Icon as={expanded ? CaretDownIcon : CaretRightIcon} size={12} />
                          <span className={a.app == null || a.app === "anonymous" ? "us-app-bucket" : undefined}>
                            {label}
                          </span>
                          <span className="us-app-count">{t("apps.recipes", { count: a.recipes.length })}</span>
                        </button>
                      </td>
                      <FigureCells r={a} f={f} estimate={estimate} />
                    </tr>
                    {expanded &&
                      a.recipes.map((r) => (
                        <tr key={`${k}/${r.modality}/${r.recipe}`} className="us-app-sub">
                          <td>
                            <ModelLabel worker={r.modality} model={r.recipe} showWorker />
                          </td>
                          <FigureCells r={r} f={f} estimate={estimate} />
                        </tr>
                      ))}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
