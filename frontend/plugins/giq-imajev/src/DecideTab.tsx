// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { api, components, sandbox, useFormat, useRunner, type PanelProps } from "@giq/plugin-ui";
import { useTranslation } from "react-i18next";
import { AnswerCard } from "./AnswerCard";
import type { DecideAnswer } from "./shapes";
import { JsonEditor } from "./JsonEditor";
import { NS } from "./ns";
import "./AnswerCard.css";
import "./DecideTab.css";
import "./JsonEditor.css";

export type DecideTabProps = PanelProps;

interface DecideResponse {
  answers: Record<string, DecideAnswer>;
  tokens_in?: number | null;
  images?: number | null;
  rotations?: number | null;
  job_id?: string;
}

/* Typed classification: state + questions JSON plus 0–2 images, straight at
   POST /decide (the Jev contract with photos). One flat section per question
   in the result card, shaped by its type (AnswerCard): the picked option,
   score or verdict, probability bars, and the trained can't-tell as a
   footnote or an abstained note. */
export function DecideTab({ options, model, onModel }: PanelProps) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const runner = useRunner<DecideResponse>();
  const [stateText, setStateText] = useState(() => t("decide.defaultState"));
  const [questionsText, setQuestionsText] = useState(() => t("decide.defaultQuestions"));
  const [file0, setFile0] = useState<File | null>(null);
  const [file1, setFile1] = useState<File | null>(null);
  const [jsonError, setJsonError] = useState<string | null>(null);
  const noWeights = options.find((o) => o.model === model)?.onDisk === false;

  const go = () => {
    let payload: { state?: unknown; questions: unknown };
    try {
      // State is the record the questions read: a JSON value, or "" for none.
      // A "quoted string" stays a string; only {…} / […] must parse as JSON.
      const raw = stateText.trim();
      const state =
        raw === "" ? {} : raw.startsWith("{") || raw.startsWith("[") ? (JSON.parse(raw) as unknown) : raw;
      const questions = JSON.parse(questionsText) as unknown;
      if (!questions || typeof questions !== "object" || Array.isArray(questions)) throw new Error("questions");
      payload = { state, questions };
    } catch {
      setJsonError(t("decide.badJson"));
      return;
    }
    setJsonError(null);
    void runner.run(async (signal) => {
      if (file0 || file1) {
        const form = new FormData();
        form.set("request", JSON.stringify(payload));
        if (file0) form.append("image0", file0);
        if (file1) form.append("image1", file1);
        return api.postForm<DecideResponse>(`/decide?model=${encodeURIComponent(model)}`, form, { signal });
      }
      return runDecideJob(model, payload, [], { signal });
    });
  };

  const answers = runner.result ? Object.entries(runner.result.answers) : [];
  const warning = !options.length
    ? t("decide.none")
    : noWeights
      ? t("decide.noWeights", { model })
      : null;

  return (
    <div className="sbx-panel">
      <div className="card elev-sm sbx-form">
        <components.Field label={t("field.model")}>
          {(id) => <sandbox.ModelSelect id={id} options={options} value={model} onChange={onModel} showDisk />}
        </components.Field>
        <div className="sbx-row">
          <components.Field label={t("decide.imageRef")} className="sbx-grow">
            {(id) => <components.FilePicker id={id} accept="image/*" onChange={setFile0} />}
          </components.Field>
          <components.Field label={t("decide.imageTarget")} className="sbx-grow">
            {(id) => <components.FilePicker id={id} accept="image/*" onChange={setFile1} />}
          </components.Field>
        </div>
        <components.Field label={t("decide.state")} error={jsonError} hintTop={t("decide.stateHint")}>
          {(id) => <JsonEditor id={id} value={stateText} onChange={setStateText} rows={4} />}
        </components.Field>
        <components.Field label={t("decide.questions")} hintTop={t("decide.questionsHint")}>
          {(id) => <JsonEditor id={id} value={questionsText} onChange={setQuestionsText} rows={10} />}
        </components.Field>
        {warning && <p className={noWeights || !options.length ? "warn" : "hint"}>{warning}</p>}
        <div className="sbx-actions">
          <sandbox.RunButton
            busy={runner.busy}
            label={t("run.decide")}
            busyLabel={t("run.deciding")}
            disabled={!model || noWeights}
            onClick={go}
          />
        </div>
      </div>
      {runner.started && (
        <sandbox.OutputCard
          busy={runner.busy}
          error={runner.error ?? jsonError}
          status={
            runner.busy
              ? t("run.deciding")
              : runner.ms != null
                ? t("out.latency", { value: f.dur(runner.ms) })
                : null
          }
        >
          {runner.result && !runner.busy && (
            <>
              <ol className="pl-giq-imajev-answers">
                {answers.map(([key, a]) => (
                  <AnswerCard key={key} name={key} answer={a} />
                ))}
              </ol>
              <p className="hint">
                {t("decide.tokens", { n: runner.result.tokens_in ?? 0 })} ·{" "}
                {t("decide.job", { id: runner.result.job_id ?? "" })}
              </p>
            </>
          )}
        </sandbox.OutputCard>
      )}
    </div>
  );
}


export async function runDecideJob(
  model: string,
  payload: { state?: unknown; questions: unknown },
  images: string[],
  opts: { signal?: AbortSignal } = {},
): Promise<DecideResponse> {
  return api.postJSON<DecideResponse>(
    `/decide?model=${encodeURIComponent(model)}`,
    {
      ...payload,
      ...Object.fromEntries(images.slice(0, 2).map((b64, i) => [`image${i}_b64`, b64])),
    },
    opts,
  );
}
