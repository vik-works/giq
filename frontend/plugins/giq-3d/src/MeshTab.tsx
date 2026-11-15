// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { MeshViewer } from "./MeshViewer";
import { runMeshJob, type MeshResult } from "./meshJob";
import { FACES, MAX_CLOSE, NO_OPTIONS, optionFields, TEXTURE_SIZE, TEXTURE_SIZES, type MeshMode } from "./meshTask";
import { useObjectUrl } from "./useObjectUrl";
import { NS } from "./ns";
import "./MeshTab.css";

const { Field, FilePicker, SegmentedControl } = components;
const { ModelSelect, OutputCard, RunButton, fileB64 } = sandbox;

const ACCEPT = "image/png,image/jpeg,image/webp";
/* The order a multi-view recipe takes its views in: a quarter turn each. */
const VIEWS = ["front", "right", "back", "left"] as const;

export function MeshTab({ options, model, onModel }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const [input, setInput] = useState<"picture" | "views">("picture");
  const [file, setFile] = useState<File | null>(null);
  const [views, setViews] = useState<(File | null)[]>([null, null, null, null]);
  const [missing, setMissing] = useState(false);
  const [opts, setOpts] = useState(NO_OPTIONS);
  const runner = useRunner<MeshResult>();
  const url = useObjectUrl(runner.result?.glb);
  const fit = options.find((o) => o.model === model)?.fits;
  const faces = FACES[opts.mode];

  const go = () => {
    if (input === "picture" ? !file : views.some((v) => !v)) return setMissing(true);
    void runner.run(async (signal) => {
      const pictures =
        input === "picture"
          ? { image_b64: await fileB64(file as File) }
          : { views_b64: await Promise.all(views.map((v) => fileB64(v as File))) };
      return runMeshJob(model, { id: "sbx-mesh", ...pictures, ...optionFields(opts) }, signal);
    });
  };

  const res = runner.result;
  const meta = res && [
    res.faces != null && t("out.faces", { count: res.faces, formatted: f.num(res.faces) }),
    res.textureSize != null && t("out.texture", { size: res.textureSize }),
    t("out.size", { value: f.bytes(res.glb.size) }),
    res.seed != null && t("out.seed", { seed: res.seed }),
  ];
  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <div className="field form-field">
          <span className="form-label" aria-hidden>{t("field.input")}</span>
          <SegmentedControl
            label={t("field.input")}
            value={input}
            options={[
              { value: "picture", label: t("input.picture") },
              { value: "views", label: t("input.views") },
            ]}
            onChange={(v) => {
              setInput(v as "picture" | "views");
              setMissing(false);
            }}
          />
        </div>
        {input === "picture" ? (
          <>
            <Field label={t("field.image")} error={missing && !file ? t("mesh.needImage") : null}>
              {(id) => (
                <FilePicker
                  id={id}
                  accept={ACCEPT}
                  invalid={missing && !file}
                  onChange={(next) => {
                    setFile(next);
                    setMissing(false);
                  }}
                />
              )}
            </Field>
            <p className="hint">{t("mesh.imageHint")}</p>
          </>
        ) : (
          <>
            <div className="pl-giq-3d-views">
              {VIEWS.map((name, i) => (
                <Field key={name} label={t(`view.${name}`)} error={missing && !views[i] ? t("mesh.needView") : null}>
                  {(id) => (
                    <FilePicker
                      id={id}
                      accept={ACCEPT}
                      invalid={missing && !views[i]}
                      onChange={(next) => {
                        setViews((all) => all.map((v, j) => (j === i ? next : v)));
                        setMissing(false);
                      }}
                    />
                  )}
                </Field>
              ))}
            </div>
            <p className="hint">{t("mesh.viewsHint")}</p>
          </>
        )}
        <div className="sbx-row">
          <Field label={t("field.model")} className="sbx-grow">
            {(id) => <ModelSelect id={id} options={options} value={model} onChange={onModel} />}
          </Field>
          <div className="field form-field">
            <span className="form-label" aria-hidden>{t("field.mode")}</span>
            <SegmentedControl
              label={t("field.mode")}
              value={opts.mode}
              options={[
                { value: "game", label: t("mode.game") },
                { value: "full", label: t("mode.full") },
              ]}
              // The two exports count triangles on different scales, so a
              // number typed for one would be out of range for the other.
              onChange={(v) => setOpts((o) => ({ ...o, mode: v as MeshMode, faces: "" }))}
            />
          </div>
        </div>
        <p className="hint">{t(opts.mode === "game" ? "mesh.gameHint" : "mesh.fullHint")}</p>
        <div className="sbx-row">
          <Field label={t("field.faces")}>
            {(id) => (
              <input
                id={id}
                className="input"
                type="number"
                min={faces.min}
                max={faces.max}
                step={100}
                placeholder={f.num(faces.fallback)}
                value={opts.faces}
                onChange={(e) => setOpts((o) => ({ ...o, faces: e.target.value }))}
              />
            )}
          </Field>
          <Field label={t("field.textureSize")}>
            {(id) => (
              <select
                id={id}
                className="input"
                value={opts.textureSize || String(TEXTURE_SIZE[opts.mode])}
                onChange={(e) => setOpts((o) => ({ ...o, textureSize: e.target.value }))}
              >
                {TEXTURE_SIZES.map((size) => (
                  <option key={size} value={size}>{size}</option>
                ))}
              </select>
            )}
          </Field>
          {opts.mode === "game" && (
            <Field label={t("field.close")}>
              {(id) => (
                <input
                  id={id}
                  className="input sbx-narrow"
                  type="number"
                  min={0}
                  max={MAX_CLOSE}
                  placeholder="4"
                  value={opts.close}
                  onChange={(e) => setOpts((o) => ({ ...o, close: e.target.value }))}
                />
              )}
            </Field>
          )}
          <Field label={t("field.seed")}>
            {(id) => (
              <input
                id={id}
                className="input sbx-narrow"
                type="number"
                min={0}
                placeholder="42"
                value={opts.seed}
                onChange={(e) => setOpts((o) => ({ ...o, seed: e.target.value }))}
              />
            )}
          </Field>
        </div>
        {!options.length && <p className="warn">{t("mesh.none")}</p>}
        {fit === "fits_after_eviction" && <p className="warn">{t("mesh.evicts")}</p>}
        <div className="sbx-actions">
          <RunButton busy={runner.busy} label={t("run.mesh")} busyLabel={t("run.generating")} disabled={!model} onClick={go} />
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
          {res && url && !runner.busy && !runner.error && (
            <>
              <MeshViewer glb={res.glb} label={t("out.viewerLabel")} />
              <div className="pl-giq-3d-below">
                <span className="hint">{t("out.viewerHint")}</span>
                <a className="btn btn-sm" href={url} download="asset.glb">{t("out.download")}</a>
              </div>
            </>
          )}
        </OutputCard>
      )}
    </div>
  );
}
