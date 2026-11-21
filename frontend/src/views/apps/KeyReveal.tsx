// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { WarningIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import type { AppCreated } from "../../api/types";
import { Card, CopyButton, Icon } from "../../components";
import { keySnippets } from "./appsModel";
import "./KeyReveal.css";

/* The new app's key, the one time it exists outside the app: giq keeps
   only its hash, so this card is the last chance to copy it. It stays
   until the operator dismisses it or leaves the page. Below, where the key
   goes: an OpenAI SDK's api_key, which it sends as a Bearer header. */
export function KeyReveal({ app, onDone }: { app: AppCreated; onDone: () => void }) {
  const { t } = useTranslation("apps");
  const snippets = keySnippets(window.location.origin, app.key);
  return (
    <Card kicker={t("key.kicker", { name: app.name })} className="ap-key">
      <p className="ap-key-warn" role="alert">
        <Icon as={WarningIcon} size={14} />
        {t("key.once")}
      </p>
      <div className="ap-key-box">
        <code className="mono ap-key-value">{app.key}</code>
        <CopyButton text={app.key} label={t("key.copy")} />
      </div>
      <p className="hint">{t("key.howTo")}</p>
      <div className="ap-key-snippet">
        <pre className="mono">{snippets.python}</pre>
        <CopyButton text={snippets.python} />
      </div>
      <div className="ap-key-snippet">
        <pre className="mono">{snippets.curl}</pre>
        <CopyButton text={snippets.curl} />
      </div>
      <div className="ap-key-actions">
        <button type="button" className="btn btn-sm" onClick={onDone}>
          {t("key.done")}
        </button>
      </div>
    </Card>
  );
}
