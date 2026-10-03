// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { CheckIcon, CopyIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Icon } from "../../components/Icon";
import "./CopyCommand.css";

/* A command for the operator to run on the server, with a copy button. The
   dashboard does not run it: installing a plugin changes giq's own
   environment, and a hardened service may not write its model store. The
   copy needs a secure context (localhost or https); elsewhere the command
   is still there to select by hand. */
export function CopyCommand({ command }: { command: string }) {
  const { t } = useTranslation("recipes");
  const [copied, setCopied] = useState(false);
  const canCopy = typeof navigator !== "undefined" && !!navigator.clipboard && window.isSecureContext;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };
  return (
    <div className="rc-add-command">
      <code className="mono">{command}</code>
      {canCopy && (
        <button
          type="button"
          className="btn btn-icon btn-ghost btn-sm"
          onClick={() => void copy()}
          title={copied ? t("add.copied") : t("add.copy")}
          aria-label={copied ? t("add.copied") : t("add.copy")}
        >
          <Icon as={copied ? CheckIcon : CopyIcon} size={14} />
        </button>
      )}
    </div>
  );
}
