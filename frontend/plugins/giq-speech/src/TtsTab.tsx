// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { TTS_DEFAULT_VOICE, TTS_FALLBACK, TTS_VOICES } from "./constants";
import { NS } from "./ns";
import "./TtsTab.css";

const { Field } = components;
const { OutputCard, RunButton } = sandbox;

export function TtsTab({ options, model }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [text, setText] = useState(() => t("tts.defaultText"));
  const [voice, setVoice] = useState<string>(TTS_DEFAULT_VOICE);
  const runner = useRunner<Blob>();
  const [url, setUrl] = useState<string | null>(null);

  // One object URL per result, released when the next replaces it.
  useEffect(() => {
    if (!runner.result) return;
    const u = URL.createObjectURL(runner.result);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [runner.result]);

  const recipe = model || TTS_FALLBACK;
  const size = options.find((o) => o.model === recipe)?.vram_gb ?? null;
  const go = () =>
    void runner.run((signal) => api.postBlob("/v1/audio/speech", { input: text, voice, model: recipe }, { signal }));

  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <Field label={t("field.text")}>
          {(id) => <textarea id={id} className="input" value={text} onChange={(e) => setText(e.target.value)} />}
        </Field>
        <Field label={t("field.voice")} className="sbx-narrow">
          {(id) => (
            <select id={id} className="input" value={voice} onChange={(e) => setVoice(e.target.value)}>
              {TTS_VOICES.map((v) => (
                <option key={v} value={v}>
                  {v === TTS_DEFAULT_VOICE ? t("tts.defaultVoice", { voice: v }) : v}
                </option>
              ))}
            </select>
          )}
        </Field>
        <p className="hint">{t("tts.hint", { model: recipe, size: f.gb(size) })}</p>
        <div className="sbx-actions">
          <RunButton busy={runner.busy} label={t("run.tts")} busyLabel={t("run.synthesizing")} onClick={go} />
        </div>
      </div>
      {runner.started && (
        <OutputCard
          busy={runner.busy}
          error={runner.error}
          status={runner.busy ? t("run.synthesizing") : runner.ms != null ? t("out.latency", { value: f.dur(runner.ms) }) : null}
        >
          {url && !runner.busy && !runner.error && <audio className="pl-giq-speech-audio" controls src={url} />}
        </OutputCard>
      )}
    </div>
  );
}
