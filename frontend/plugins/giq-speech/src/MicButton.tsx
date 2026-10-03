// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { MicrophoneIcon, StopIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import { components, useFormat } from "@giq/plugin-ui";
import { NS } from "./ns";
import type { Recorder } from "./useRecorder";

const { Icon } = components;

/** Record / stop, with the size of what was captured. */
export function MicButton({ rec }: { rec: Recorder }) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  return (
    <span className="pl-giq-speech-mic">
      <button type="button" className="btn btn-secondary" aria-pressed={rec.recording} onClick={rec.toggle}>
        <Icon as={rec.recording ? StopIcon : MicrophoneIcon} size={14} />
        {rec.recording ? t("asr.stopRecording") : t("asr.record")}
      </button>
      {rec.recording && <span className="pl-giq-speech-rec-dot" aria-hidden />}
      {rec.blob && !rec.recording && (
        <span className="hint">{t("asr.captured", { size: f.bytes(rec.blob.size) })}</span>
      )}
    </span>
  );
}
