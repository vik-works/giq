// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { useAudioUrl } from "../../giq-music/src/useAudioUrl";
import { runSfxJob, type SfxResult } from "./sfxJob";
import { sfxTask, type SfxForm } from "./sfxTask";
import { NS } from "./ns";
import "./SfxTab.css";

const { Field } = components;
const { ModelSelect, OutputCard, RunButton } = sandbox;

type NumberKey = Exclude<keyof SfxForm, "prompt">;

export function SfxTab({ options, model, onModel }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [form, setForm] = useState<SfxForm>(() => ({
    prompt: t("sfx.defaultPrompt"),
    duration: "5",
    steps: "",
    guidance: "",
    seed: "",
  }));
  const runner = useRunner<SfxResult>();
  const url = useAudioUrl(runner.result?.audio);
  const fit = options.find((o) => o.model === model)?.fits;
  const set = (key: keyof SfxForm, value: string) => setForm((prev) => ({ ...prev, [key]: value }));

  const go = () => {
    const task = sfxTask(form);
    void runner.run((signal) => runSfxJob(model, task, signal));
  };

  /* A number field; an empty one leaves the value to the recipe (steps,
     guidance) or the model (seed). */
  const numberField = (key: NumberKey, attrs: { min: number; max?: number; step: number }, placeholder?: string): ReactNode => (
    <Field label={t(`field.${key}`)}>
      {(id) => (
        <input
          id={id}
          className="input pl-giq-sfx-number"
          type="number"
          {...attrs}
          placeholder={placeholder}
          value={form[key]}
          onChange={(e) => set(key, e.target.value)}
        />
      )}
    </Field>
  );

  const res = runner.result;
  const meta = res && [
    res.duration != null && t("out.duration", { value: f.dur(res.duration * 1000) }),
    res.steps != null && t("out.steps", { count: res.steps }),
    res.guidance != null && t("out.guidance", { value: f.num(res.guidance, 1) }),
    res.seed != null && t("out.seed", { seed: res.seed }),
  ];
  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <Field label={t("field.prompt")}>
          {(id) => (
            <textarea
              id={id}
              className="input pl-giq-sfx-prompt"
              value={form.prompt}
              onChange={(e) => set("prompt", e.target.value)}
            />
          )}
        </Field>
        <p className="hint">{t("sfx.promptHint")}</p>
        <div className="sbx-row">
          <Field label={t("field.model")} className="sbx-grow">
            {(id) => <ModelSelect id={id} options={options} value={model} onChange={onModel} />}
          </Field>
          {numberField("duration", { min: 0.5, max: 30, step: 0.5 })}
          {numberField("steps", { min: 1, max: 200, step: 1 }, t("field.fromRecipe"))}
          {numberField("guidance", { min: 1, max: 20, step: 0.5 }, t("field.fromRecipe"))}
          {numberField("seed", { min: 0, step: 1 }, t("field.seedRandom"))}
        </div>
        <p className="hint">{t("sfx.costHint")}</p>
        {!options.length && <p className="warn">{t("sfx.none")}</p>}
        {fit === "fits_after_eviction" && <p className="warn">{t("sfx.evicts")}</p>}
        <div className="sbx-actions">
          <RunButton
            busy={runner.busy}
            label={t("run.sfx")}
            busyLabel={t("run.rendering")}
            disabled={!model || !form.prompt.trim()}
            onClick={go}
          />
        </div>
      </div>
      {runner.started && (
        <OutputCard
          busy={runner.busy}
          error={runner.error}
          status={
            runner.busy
              ? t("run.rendering")
              : res && runner.ms != null
                ? [t("out.latency", { value: f.dur(runner.ms) }), ...(meta ?? [])].filter(Boolean).join(" · ")
                : null
          }
        >
          {res && url && !runner.busy && !runner.error && (
            <>
              <audio className="pl-giq-sfx-audio" controls src={url} />
              <div className="pl-giq-sfx-below">
                <a className="btn btn-sm" href={url} download={`sound-effect.${res.format}`}>
                  {t("out.download", { format: res.format.toUpperCase() })}
                </a>
              </div>
            </>
          )}
        </OutputCard>
      )}
    </div>
  );
}
