// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { GeneratedImage } from "./GeneratedImage";
import { ImageOptions } from "./ImageOptions";
import { runImageJob, type ImageResult } from "./imageJob";
import { NO_OPTIONS, optionFields } from "./imageTask";
import { NS } from "./ns";

const { Field } = components;
const { ModelSelect, OutputCard, RunButton } = sandbox;

export function T2iTab({ options, model, onModel }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [prompt, setPrompt] = useState(() => t("t2i.defaultPrompt"));
  const [opts, setOpts] = useState(NO_OPTIONS);
  const runner = useRunner<ImageResult>();
  const fit = options.find((o) => o.model === model)?.fits;

  const go = () => {
    const task = { id: "sbx-t2i", prompt, ...optionFields(opts) };
    void runner.run((signal) => runImageJob({ modality: "text2image", model, tasks: [task] }, signal));
  };

  const res = runner.result;
  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <Field label={t("field.prompt")}>
          {(id) => <textarea id={id} className="input" value={prompt} onChange={(e) => setPrompt(e.target.value)} />}
        </Field>
        <Field label={t("field.model")}>
          {(id) => <ModelSelect id={id} options={options} value={model} onChange={onModel} />}
        </Field>
        <ImageOptions value={opts} onChange={setOpts} sizeHint={t("t2i.sizeHint")} />
        {!options.length && <p className="warn">{t("t2i.none")}</p>}
        {fit === "fits_after_eviction" && <p className="warn">{t("t2i.evicts")}</p>}
        <div className="sbx-actions">
          <RunButton busy={runner.busy} label={t("run.t2i")} busyLabel={t("run.generatingImage")} disabled={!model} onClick={go} />
        </div>
      </div>
      {runner.started && (
        <OutputCard
          busy={runner.busy}
          error={runner.error}
          status={
            runner.busy
              ? t("run.generatingImage")
              : res && runner.ms != null
                ? [t("out.latency", { value: f.dur(runner.ms) }), res.seed != null && t("out.seed", { seed: res.seed })]
                    .filter(Boolean)
                    .join(" · ")
                : null
          }
        >
          {res && !runner.busy && <GeneratedImage b64={res.b64} alt={prompt} />}
        </OutputCard>
      )}
    </div>
  );
}
