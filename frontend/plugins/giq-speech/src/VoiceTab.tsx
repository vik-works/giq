// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api, components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import type { AudioEmbedding } from "../../../src/api/types";
import { cosine } from "./cosine";
import { CosineResult } from "./CosineResult";
import { NS } from "./ns";

const { Field, FilePicker } = components;
const { OutputCard, RunButton } = sandbox;

async function embed(file: File, model: string, signal: AbortSignal): Promise<number[]> {
  const fd = new FormData();
  fd.append("file", file, file.name);
  if (model) fd.append("model", model);
  return (await api.postForm<AudioEmbedding>("/v1/audio/embeddings", fd, { signal })).embedding;
}

export function VoiceTab({ model }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [a, setA] = useState<File | null>(null);
  const [b, setB] = useState<File | null>(null);
  const [missing, setMissing] = useState(false);
  const runner = useRunner<number>();

  const go = () => {
    if (!a || !b) return setMissing(true);
    void runner.run(async (signal) => {
      const [ea, eb] = await Promise.all([embed(a, model, signal), embed(b, model, signal)]);
      return cosine(ea, eb);
    });
  };

  const clip = (label: string, file: File | null, set: (f: File | null) => void) => (
    <Field label={label} className="sbx-grow" error={missing && !file ? t("voice.needClip") : null}>
      {(id) => (
        <FilePicker
          id={id}
          accept="audio/*"
          invalid={missing && !file}
          onChange={(next) => {
            set(next);
            setMissing(false);
          }}
        />
      )}
    </Field>
  );

  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <div className="sbx-row">
          {clip(t("voice.clipA"), a, setA)}
          {clip(t("voice.clipB"), b, setB)}
        </div>
        <p className="hint">{t("voice.hint")}</p>
        <div className="sbx-actions">
          <RunButton busy={runner.busy} label={t("run.voice")} busyLabel={t("run.embedding")} onClick={go} />
        </div>
      </div>
      {runner.started && (
        <OutputCard
          busy={runner.busy}
          error={runner.error}
          status={runner.busy ? t("run.embedding") : runner.ms != null ? t("out.latency", { value: f.dur(runner.ms) }) : null}
        >
          {runner.result != null && !runner.busy && <CosineResult cos={runner.result} />}
        </OutputCard>
      )}
    </div>
  );
}
