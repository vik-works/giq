// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { GeneratedImage } from "./GeneratedImage";
import { ImageOptions } from "./ImageOptions";
import { runImageJob, type ImageResult } from "./imageJob";
import { imageSize, NO_OPTIONS, optionFields } from "./imageTask";
import { NS } from "./ns";

const { Field, FilePicker } = components;
const { ModelSelect, OutputCard, RunButton, fileB64 } = sandbox;

export function EditTab({ options, model, onModel }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [file, setFile] = useState<File | null>(null);
  const [missing, setMissing] = useState(false);
  const [instruction, setInstruction] = useState(() => t("edit.defaultPrompt"));
  const [opts, setOpts] = useState(NO_OPTIONS);
  const runner = useRunner<ImageResult>();
  const fit = options.find((o) => o.model === model)?.fits;

  const go = () => {
    if (!file) return setMissing(true);
    void runner.run(async (signal) =>
      runImageJob(
        {
          modality: "image_edit",
          model,
          tasks: [{ id: "sbx-edit", reference_image_b64: await fileB64(file), instruction, ...optionFields(opts) }],
        },
        signal,
      ),
    );
  };

  const res = runner.result;
  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <Field label={t("field.model")}>
          {(id) => <ModelSelect id={id} options={options} value={model} onChange={onModel} />}
        </Field>
        <Field label={t("field.referenceImage")} error={missing && !file ? t("edit.needImage") : null}>
          {(id) => (
            <FilePicker
              id={id}
              accept="image/*"
              invalid={missing && !file}
              onChange={(next) => {
                setFile(next);
                setMissing(false);
                // The edit keeps the reference's shape unless asked otherwise:
                // the server's default size would squeeze a portrait into 800x480.
                if (next)
                  void imageSize(next).then((size) =>
                    setOpts((o) => (size ? { ...o, width: String(size.width), height: String(size.height) } : o)),
                  );
              }}
            />
          )}
        </Field>
        <Field label={t("field.instruction")}>
          {(id) => <textarea id={id} className="input" value={instruction} onChange={(e) => setInstruction(e.target.value)} />}
        </Field>
        <ImageOptions value={opts} onChange={setOpts} sizeHint={t("edit.sizeHint")} />
        {!options.length && <p className="warn">{t("edit.none")}</p>}
        {fit === "fits_after_eviction" && <p className="warn">{t("edit.evicts")}</p>}
        <div className="sbx-actions">
          <RunButton busy={runner.busy} label={t("run.edit")} busyLabel={t("run.editing")} disabled={!model} onClick={go} />
        </div>
      </div>
      {runner.started && (
        <OutputCard
          busy={runner.busy}
          error={runner.error}
          status={
            runner.busy
              ? t("run.editing")
              : res && runner.ms != null
                ? [t("out.latency", { value: f.dur(runner.ms) }), res.seed != null && t("out.seed", { seed: res.seed })]
                    .filter(Boolean)
                    .join(" · ")
                : null
          }
        >
          {res && !runner.busy && <GeneratedImage b64={res.b64} alt={instruction} />}
        </OutputCard>
      )}
    </div>
  );
}
