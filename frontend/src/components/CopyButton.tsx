// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { CheckIcon, CopyIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Icon } from "./Icon";

/* A button that copies a text and says so for a moment. The clipboard needs
   a secure context (localhost or https); elsewhere the button is left out,
   and the text beside it stays there to select by hand. */
export function CopyButton({ text, label }: { text: string; label?: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const canCopy = typeof navigator !== "undefined" && !!navigator.clipboard && window.isSecureContext;
  if (!canCopy) return null;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };
  const name = copied ? t("actions.copied") : (label ?? t("actions.copy"));
  return (
    <button
      type="button"
      className="btn btn-icon btn-ghost btn-sm"
      onClick={() => void copy()}
      title={name}
      aria-label={name}
    >
      <Icon as={copied ? CheckIcon : CopyIcon} size={14} />
    </button>
  );
}
