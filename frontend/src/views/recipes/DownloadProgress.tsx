// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { XIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import type { Download } from "../../api/types";
import { Icon } from "../../components/Icon";
import { ProgressBar } from "../../components/ProgressBar";
import { useFormat } from "../../lib/useFormat";
import "./DownloadProgress.css";

export interface DownloadProgressProps {
  d: Download;
  onCancel: () => void;
}

/* A running or queued fetch: how far, of how much, and a way to stop it.
   The total is the Hub's figure from the plan; when the Hub did not answer
   it is unknown and only the bytes so far are shown. */
export function DownloadProgress({ d, onCancel }: DownloadProgressProps) {
  const { t } = useTranslation("recipes");
  const fmt = useFormat();
  const known = d.bytes_total > 0;
  const text = known
    ? t("add.progress", { done: fmt.bytes(d.bytes_done), total: fmt.bytes(d.bytes_total) })
    : t("add.progressUnknown", { done: fmt.bytes(d.bytes_done) });
  return (
    <div className="rc-add-progress">
      <div className="rc-add-progress-line">
        <span>{d.state === "queued" ? t("add.queued") : text}</span>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={onCancel}
          title={t("add.cancelHelp")}
        >
          <Icon as={XIcon} size={12} />
          {t("add.cancel")}
        </button>
      </div>
      <ProgressBar
        value={known ? d.bytes_done : 0}
        max={known ? d.bytes_total : 1}
        label={t("add.progressLabel", { recipe: d.recipe })}
        valueText={text}
      />
      {d.current && <span className="rc-add-progress-from mono">{d.current}</span>}
    </div>
  );
}
