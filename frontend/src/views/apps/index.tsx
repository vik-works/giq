// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { AppCreated } from "../../api/types";
import { PageHeader } from "../../components";
import { useApps } from "../../state";
import { AppsList } from "./AppsList";
import { CreateAppForm } from "./CreateAppForm";
import { KeyReveal } from "./KeyReveal";
import "./AppsView.css";

/* Apps (ADR-006): a name and a key, so the Usage view can say which app
   used what. The page creates one, shows its key once, and revokes or
   deletes. The key lives in this component's state only: leaving the page
   drops it, as nothing can fetch it again. */
export default function AppsView() {
  const { t } = useTranslation("apps");
  const apps = useApps();
  const [created, setCreated] = useState<AppCreated | null>(null);
  const onCreated = (app: AppCreated) => {
    setCreated(app);
    void apps.refresh();
  };
  return (
    <>
      <PageHeader title={t("common:nav.apps")} />
      <p className="hint ap-intro">{t("intro")}</p>
      <div className="ap-layout">
        <div className="ap-main stack">
          {created && <KeyReveal app={created} onDone={() => setCreated(null)} />}
          <AppsList />
        </div>
        <aside className="ap-side">
          <CreateAppForm onCreated={onCreated} />
        </aside>
      </div>
    </>
  );
}
