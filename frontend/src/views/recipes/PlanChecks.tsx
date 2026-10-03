// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { CheckCircleIcon, WarningIcon, XCircleIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import type { PlanCheck } from "../../api/types";
import { Icon } from "../../components/Icon";
import "./PlanChecks.css";

const GLYPH = { ok: CheckCircleIcon, warn: WarningIcon, fail: XCircleIcon } as const;

/* The plan's checks, one line each: an icon and a word for the status (never
   colour alone), and the server's message as written — it names the file,
   the size or the command, which a translated paraphrase would lose. */
export function PlanChecks({ checks }: { checks: PlanCheck[] }) {
  const { t } = useTranslation("recipes");
  return (
    <ul className="rc-add-checks">
      {checks.map((c, i) => (
        <li key={`${c.check}-${i}`} className={`rc-add-check rc-add-check-${c.status}`}>
          <Icon as={GLYPH[c.status]} size={14} label={t(`add.status.${c.status}`)} />
          <span className="rc-add-check-name">{t(`add.checkName.${c.check}`, { defaultValue: c.check })}</span>
          <span className="rc-add-check-message">{c.message}</span>
        </li>
      ))}
    </ul>
  );
}
