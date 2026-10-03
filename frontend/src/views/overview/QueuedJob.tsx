// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { XIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import { Icon } from "../../components/Icon";
import { WorkerIcon } from "../../components/WorkerIcon";
import type { JobInfo } from "./useJobDetails";
import { useModalityLabel } from "../../lib/modalities";

export interface QueuedJobProps {
  id: string;
  /** 1-based queue position. */
  pos: number;
  info: JobInfo | null | undefined;
  running: boolean;
  onCancel?: () => void;
  cancelling?: boolean;
}

/** One job in the queue card: what it is, where it stands, and (queued only) a way to cancel it. */
export function QueuedJob({ id, pos, info, running, onCancel, cancelling }: QueuedJobProps) {
  const { t } = useTranslation("overview");
  const modalityLabel = useModalityLabel();
  const what = info ? (
    <>
      <WorkerIcon worker={info.worker} size={running ? 14 : 13} colored={running} />
      <span className="ov-jq-model">{info.model}</span>
      <span className="ov-jq-worker">{modalityLabel(info.worker)}</span>
    </>
  ) : (
    <span className="ov-jq-worker">{info === null ? t("queue.gone") : t("queue.resolving")}</span>
  );

  if (running) {
    return (
      <li className="ov-jq-running">
        <div className="ov-jq-line" title={id}>
          {what}
          <span className="ov-jq-meta">{t("queue.running")}</span>
        </div>
        {/* giq reports no per-job progress, so the bar says "working", not a percentage it would have to invent. */}
        <div className="ov-jq-bar" role="progressbar" aria-label={t("queue.runningLabel", { id })}>
          <span />
        </div>
      </li>
    );
  }
  return (
    <li className="ov-jq-queued" title={id}>
      <span className="ov-jq-pos">{t("queue.position", { pos })}</span>
      {what}
      <button
        type="button"
        className="btn btn-icon btn-ghost btn-sm ov-jq-cancel"
        aria-label={t("queue.cancel")}
        title={t("queue.cancel")}
        disabled={cancelling}
        onClick={onCancel}
      >
        <Icon as={XIcon} size={12} />
      </button>
    </li>
  );
}
