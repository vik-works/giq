// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import { deleteApp, revokeApp } from "../../api/apps";
import { errorText } from "../../api/client";
import type { AppEntry } from "../../api/types";
import { useConfirm, useNotify } from "../../components/DialogProvider";
import { useApps } from "../../state";

/* Revoking and deleting an app, each behind a confirmation that says what
   it does to the name and the history, since the two differ only there.
   A failure is the server's text, shown as is; either way the list is
   re-read at once. */
export function useAppActions() {
  const { t } = useTranslation("apps");
  const confirm = useConfirm();
  const notify = useNotify();
  const apps = useApps();
  const [busy, setBusy] = useState<string | null>(null);

  const run = useCallback(
    async (name: string, call: () => Promise<unknown>, failed: string) => {
      setBusy(name);
      try {
        await call();
      } catch (err) {
        await notify({ title: failed, body: errorText(err) });
      } finally {
        setBusy(null);
        await apps.refresh();
      }
    },
    [notify, apps.refresh],
  );

  const revoke = useCallback(
    async (a: AppEntry) => {
      const ok = await confirm({
        title: t("revoke.confirmTitle", { name: a.name }),
        body: t("revoke.confirmBody"),
        confirmLabel: t("revoke.confirm"),
        danger: true,
      });
      if (ok) await run(a.name, () => revokeApp(a.name), t("revoke.failed"));
    },
    [t, confirm, run],
  );

  const remove = useCallback(
    async (a: AppEntry) => {
      const ok = await confirm({
        title: t("delete.confirmTitle", { name: a.name }),
        body: t("delete.confirmBody"),
        confirmLabel: t("delete.confirm"),
        danger: true,
      });
      if (ok) await run(a.name, () => deleteApp(a.name), t("delete.failed"));
    },
    [t, confirm, run],
  );

  return { busy, revoke, remove };
}
