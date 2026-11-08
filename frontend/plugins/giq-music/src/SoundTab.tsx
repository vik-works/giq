// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { runAudioJob, type MusicResult } from "./musicJob";
import { NS } from "./ns";
import { useAudioUrl } from "./useAudioUrl";
import "./MusicTab.css";

const { Field } = components;
const { ModelSelect, OutputCard, RunButton } = sandbox;

/* Text → sound: a description and a length, a clip back. */
export function SoundTab({ options, model, onModel }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [prompt, setPrompt] = useState(() => t("sound.defaultPrompt"));
  const [negative, setNegative] = useState("");
  const [duration, setDuration] = useState("10");
  const [seed, setSeed] = useState("");
  const runner = useRunner<MusicResult>();
  const url = useAudioUrl(runner.result?.audio);
  const fit = options.find((o) => o.model === model)?.fits;

  const go = () => {
    const task: Record<string, unknown> = { id: "sbx-sound", prompt };
    if (negative.trim()) task.negative_prompt = negative;
    if (duration !== "") task.duration = Number(duration);
    if (seed !== "") task.seed = Number(seed);
    void runner.run((signal) => runAudioJob("text2sfx", model, task, signal));
  };

  const res = runner.result;
  const meta = res && [
    res.duration != null && t("out.duration", { value: f.dur(res.duration * 1000) }),
    res.seed != null && t("out.seed", { seed: res.seed }),
  ];
  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <Field label={t("field.prompt")}>
          {(id) => <textarea id={id} className="input" value={prompt} onChange={(e) => setPrompt(e.target.value)} />}
        </Field>
        <Field label={t("field.negative")}>
          {(id) => (
            <input
              id={id}
              className="input"
              placeholder={t("field.negativeHint")}
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
            />
          )}
        </Field>
        <div className="sbx-row">
          <Field label={t("field.model")} className="sbx-grow">
            {(id) => <ModelSelect id={id} options={options} value={model} onChange={onModel} />}
          </Field>
          <Field label={t("field.duration")}>
            {(id) => (
              <input id={id} className="input" type="number" min={1} max={120} value={duration} onChange={(e) => setDuration(e.target.value)} />
            )}
          </Field>
          <Field label={t("field.seed")}>
            {(id) => (
              <input id={id} className="input" type="number" placeholder={t("field.seedRandom")} value={seed} onChange={(e) => setSeed(e.target.value)} />
            )}
          </Field>
        </div>
        {!options.length && <p className="warn">{t("sound.none")}</p>}
        {fit === "fits_after_eviction" && <p className="warn">{t("sound.evicts")}</p>}
        <div className="sbx-actions">
          <RunButton busy={runner.busy} label={t("run.sound")} busyLabel={t("run.generating")} disabled={!model} onClick={go} />
        </div>
      </div>
      {runner.started && (
        <OutputCard
          busy={runner.busy}
          error={runner.error}
          status={
            runner.busy
              ? t("run.generating")
              : res && runner.ms != null
                ? [t("out.latency", { value: f.dur(runner.ms) }), ...(meta ?? [])].filter(Boolean).join(" · ")
                : null
          }
        >
          {url && !runner.busy && !runner.error && <audio className="pl-giq-music-audio" controls src={url} />}
        </OutputCard>
      )}
    </div>
  );
}
