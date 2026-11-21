// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { errorText } from "../../api/client";
import { Card, EmptyState, Tag } from "../../components";
import { DASH } from "../../lib/format";
import { useFormat } from "../../lib/useFormat";
import { useApps } from "../../state";
import { appStatus, isoSeconds, sortApps, usageHref } from "./appsModel";
import { useAppActions } from "./useAppActions";
import "./AppsList.css";

/* Every registered app, active first. A description is the operator's
   text and renders as text. Revoke stops the key and keeps the name taken;
   delete frees the name, and an app created under it again adds to the old
   history. The hint under the table says so once, the dialogs again. */
export function AppsList() {
  const { t } = useTranslation("apps");
  const f = useFormat();
  const apps = useApps();
  const { busy, revoke, remove } = useAppActions();
  const list = useMemo(() => (apps.data ? sortApps(apps.data.apps) : undefined), [apps.data]);
  return (
    <Card kicker={t("list.kicker")} actions={list ? t("list.meta", { count: list.length }) : undefined}>
      {apps.error && !list ? (
        <p className="error-box" role="alert">
          {t("common:empty.failed", { error: errorText(apps.error) })}
        </p>
      ) : !list ? (
        <EmptyState compact>{t("common:empty.loading")}</EmptyState>
      ) : list.length === 0 ? (
        <EmptyState compact>{t("list.empty")}</EmptyState>
      ) : (
        <div className="table-wrap">
          <table className="table ap-table">
            <thead>
              <tr>
                <th>{t("list.name")}</th>
                <th>{t("list.description")}</th>
                <th>{t("list.created")}</th>
                <th>{t("list.status")}</th>
                <th>
                  <span className="sr-only">{t("list.actions")}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {list.map((a) => {
                const status = appStatus(a);
                const created = isoSeconds(a.created);
                const revoked = isoSeconds(a.revoked);
                return (
                  <tr key={a.name} className={status === "revoked" ? "ap-revoked" : undefined}>
                    <td className="mono ap-name">{a.name}</td>
                    <td className="ap-desc">{a.description ?? DASH}</td>
                    <td className="ap-nowrap" title={created ? f.dateTime(created) : undefined}>
                      {f.day(created)}
                    </td>
                    <td className="ap-nowrap">
                      {status === "active" ? (
                        <Tag tone="good">{t("status.active")}</Tag>
                      ) : (
                        <Tag tone="neutral" title={revoked ? f.dateTime(revoked) : undefined}>
                          {t("status.revoked")}
                        </Tag>
                      )}
                    </td>
                    <td className="ap-actions">
                      <a className="btn btn-ghost btn-sm" href={usageHref(a.name)}>
                        {t("list.usage")}
                      </a>
                      {status === "active" && (
                        <button
                          type="button"
                          className="btn btn-sm"
                          disabled={busy === a.name}
                          onClick={() => void revoke(a)}
                        >
                          {t("revoke.action")}
                        </button>
                      )}
                      <button
                        type="button"
                        className="btn btn-sm ap-delete"
                        disabled={busy === a.name}
                        onClick={() => void remove(a)}
                      >
                        {t("delete.action")}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="hint">{t("list.hint")}</p>
    </Card>
  );
}
