// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { AudioGuide } from "./AudioGuide";
import { audioFields, isSource, NO_AUDIO } from "./audioTask";
import { runAudioJob, type MusicResult } from "./musicJob";
import { useAudioUrl } from "./useAudioUrl";
import { NS } from "./ns";
import "./MusicTab.css";

const { Field, SegmentedControl } = components;
const { ModelSelect, OutputCard, RunButton, fileB64 } = sandbox;

export function MusicTab({ options, model, onModel }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [prompt, setPrompt] = useState(() => t("music.defaultPrompt"));
  const [lyrics, setLyrics] = useState("");
  const [duration, setDuration] = useState("30");
  const [seed, setSeed] = useState("");
  const [plan, setPlan] = useState(true);
  const [audio, setAudio] = useState(NO_AUDIO);
  const runner = useRunner<MusicResult>();
  const url = useAudioUrl(runner.result?.audio);
  const chosen = options.find((o) => o.model === model);
  const fit = chosen?.fits;
  /* Lyrics and the planner are ACE-Step's; Stable Audio renders the
     description alone, and shorter or longer. */
  const songs = chosen?.engine === "acestep";
  const [minS, maxS] = songs ? [10, 600] : [1, 380];
  /* Built on the user's audio, the track is as long as it, and the planner
     sits out (ACE-Step skips it). */
  const sourced = songs && isSource(audio);

  const go = () => {
    const task: Record<string, unknown> = { id: "sbx-music", prompt };
    if (songs) {
      task.plan = plan;
      if (lyrics.trim()) task.lyrics = lyrics;
    }
    if (duration !== "" && !sourced) task.duration = Number(duration);
    if (seed !== "") task.seed = Number(seed);
    void runner.run(async (signal) => {
      if (songs) Object.assign(task, await audioFields(audio, fileB64));
      return runAudioJob("text2music", model, task, signal);
    });
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
        {songs && (
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
        )}
        {songs && <AudioGuide value={audio} onChange={setAudio} />}
        <div className="sbx-row">
          <Field label={t("field.model")} className="sbx-grow">
            {(id) => <ModelSelect id={id} options={options} value={model} onChange={onModel} />}
          </Field>
          <Field label={t("field.duration")}>
            {(id) => (
              <input
                id={id}
                className="input"
                type="number"
                min={minS}
                max={maxS}
                disabled={sourced}
                value={sourced ? "" : duration}
                placeholder={sourced ? t("field.durationOfAudio") : undefined}
                onChange={(e) => setDuration(e.target.value)}
              />
            )}
          </Field>
          <Field label={t("field.seed")}>
            {(id) => (
              <input id={id} className="input" type="number" placeholder={t("field.seedRandom")} value={seed} onChange={(e) => setSeed(e.target.value)} />
            )}
          </Field>
          {songs && (
            <div className="field form-field">
              <span className="form-label" aria-hidden>{t("field.plan")}</span>
              <SegmentedControl
                label={t("field.plan")}
                value={plan && !sourced ? "on" : "off"}
                disabled={sourced}
                options={[
                  { value: "on", label: t("toggle.on") },
                  { value: "off", label: t("toggle.off") },
                ]}
                onChange={(v) => setPlan(v === "on")}
              />
            </div>
          )}
        </div>
        <p className="hint">{t(songs ? "music.planHint" : "music.instrumentalHint")}</p>
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
