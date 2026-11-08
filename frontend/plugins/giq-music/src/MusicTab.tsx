// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { runMusicJob, type MusicResult } from "./musicJob";
import { NS } from "./ns";
import "./MusicTab.css";

const { Field, SegmentedControl } = components;
const { ModelSelect, OutputCard, RunButton } = sandbox;

export function MusicTab({ options, model, onModel }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [prompt, setPrompt] = useState(() => t("music.defaultPrompt"));
  const [lyrics, setLyrics] = useState("");
  const [duration, setDuration] = useState("30");
  const [seed, setSeed] = useState("");
  const [plan, setPlan] = useState(true);
  const runner = useRunner<MusicResult>();
  const [url, setUrl] = useState<string | null>(null);
  const fit = options.find((o) => o.model === model)?.fits;

  // One object URL per track, released when the next replaces it.
  useEffect(() => {
    if (!runner.result) return;
    const u = URL.createObjectURL(runner.result.audio);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [runner.result]);

  const go = () => {
    const task: Record<string, unknown> = { id: "sbx-music", prompt, plan };
    if (lyrics.trim()) task.lyrics = lyrics;
    if (duration !== "") task.duration = Number(duration);
    if (seed !== "") task.seed = Number(seed);
    void runner.run((signal) => runMusicJob(model, task, signal));
  };

  const res = runner.result;
  const meta = res && [
    res.duration != null && t("out.duration", { value: f.dur(res.duration * 1000) }),
    res.bpm != null && t("out.bpm", { bpm: res.bpm }),
    res.keyScale,
    res.timeSignature && t("out.timeSignature", { value: res.timeSignature }),
    res.seed != null && t("out.seed", { seed: res.seed }),
  ];
  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <Field label={t("field.prompt")}>
          {(id) => <textarea id={id} className="input" value={prompt} onChange={(e) => setPrompt(e.target.value)} />}
        </Field>
        <Field label={t("field.lyrics")}>
          {(id) => (
            <textarea
              id={id}
              className="input pl-giq-music-lyrics"
              placeholder={t("field.lyricsNone")}
              value={lyrics}
              onChange={(e) => setLyrics(e.target.value)}
            />
          )}
        </Field>
        <div className="sbx-row">
          <Field label={t("field.model")} className="sbx-grow">
            {(id) => <ModelSelect id={id} options={options} value={model} onChange={onModel} />}
          </Field>
          <Field label={t("field.duration")}>
            {(id) => (
              <input id={id} className="input" type="number" min={10} max={600} value={duration} onChange={(e) => setDuration(e.target.value)} />
            )}
          </Field>
          <Field label={t("field.seed")}>
            {(id) => (
              <input id={id} className="input" type="number" placeholder={t("field.seedRandom")} value={seed} onChange={(e) => setSeed(e.target.value)} />
            )}
          </Field>
          <div className="field form-field">
            <span className="form-label" aria-hidden>{t("field.plan")}</span>
            <SegmentedControl
              label={t("field.plan")}
              value={plan ? "on" : "off"}
              options={[
                { value: "on", label: t("toggle.on") },
                { value: "off", label: t("toggle.off") },
              ]}
              onChange={(v) => setPlan(v === "on")}
            />
          </div>
        </div>
        <p className="hint">{t("music.planHint")}</p>
        {!options.length && <p className="warn">{t("music.none")}</p>}
        {fit === "fits_after_eviction" && <p className="warn">{t("music.evicts")}</p>}
        <div className="sbx-actions">
          <RunButton busy={runner.busy} label={t("run.music")} busyLabel={t("run.composing")} disabled={!model} onClick={go} />
        </div>
      </div>
      {runner.started && (
        <OutputCard
          busy={runner.busy}
          error={runner.error}
          status={
            runner.busy
              ? t("run.composing")
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
