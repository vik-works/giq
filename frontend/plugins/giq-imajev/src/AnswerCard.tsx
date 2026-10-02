// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import {
  GaugeIcon,
  ListBulletsIcon,
  ListChecksIcon,
  ScalesIcon,
  WarningIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";
import { components, useFormat } from "@giq/plugin-ui";
import { useTranslation } from "react-i18next";
import { NS } from "./ns";
import "./AnswerCard.css";

const DASH = "\u2013";

import type { DecideAnswer } from "./shapes";

/* One symbol per answer type: the header tag names the type in words and
   carries its icon, so choice / score / noul (and multi) scan apart. */
export const TYPE_ICON: Record<string, PhosphorIcon> = {
  choice: ListBulletsIcon,
  score: GaugeIcon,
  noul: ScalesIcon,
  multi: ListChecksIcon,
};


import { noulVerdict, scoreLevels, sortedProbs } from "./shapes";
import type { ScoreLevel } from "./shapes";

/* One answer per question: the picked option, score or verdict up front, its
   distribution as shared progress bars, and the trained can't-tell as a
   footnote or an abstained note. A flat section inside the result card, the
   way the trace and the voice verdict sit flat in theirs — never a card in
   a card. */
export function AnswerCard({ name, answer: a }: { name: string; answer: DecideAnswer }) {
  const { t } = useTranslation(NS);
  return (
    <li className="pl-giq-imajev-answer">
      <div className="pl-giq-imajev-answer-head">
        <span className="pl-giq-imajev-answer-name">{name}</span>
        <components.Tag tone="neutral" icon={<components.Icon as={TYPE_ICON[a.type] ?? ListBulletsIcon} size={12} />}>
          {t(`decide.type.${a.type}`, { defaultValue: a.type })}
        </components.Tag>
      </div>
      {a.type === "score" ? (
        <ScoreBody a={a} />
      ) : a.type === "noul" ? (
        <NoulBody a={a} />
      ) : a.type === "multi" ? (
        <MultiBody a={a} />
      ) : (
        <ChoiceBody a={a} />
      )}
    </li>
  );
}

function ChoiceBody({ a }: { a: DecideAnswer }) {
  const probs = sortedProbs(a);
  if (a.abstained) {
    return (
      <>
        <div className="pl-giq-imajev-answer-value">{DASH}</div>
        <AbstainedNote />
      </>
    );
  }
  return (
    <>
      <div className="pl-giq-imajev-answer-value">{a.choice ?? DASH}</div>
      <ProbRows
        rows={probs.map(([label, p]) => ({ label, p }))}
        winners={new Set(a.choice != null ? [a.choice] : [])}
      />
      <Footnotes confidence={a.confidence} unknown={a.unknown_probability ?? 0} />
    </>
  );
}

function ScoreBody({ a }: { a: DecideAnswer }) {
  const f = useFormat();
  const levels = scoreLevels(a);
  const top = levels.length ? levels[levels.length - 1] : undefined;
  const max = top && Number.isFinite(Number(top.level)) ? Number(top.level) : null;
  let nearest: ScoreLevel | undefined;
  if (a.score != null) {
    for (const l of levels) {
      if (nearest == null || Math.abs(Number(l.level) - a.score) < Math.abs(Number(nearest.level) - a.score)) {
        nearest = l;
      }
    }
  }
  const modal = levels.reduce<ScoreLevel | undefined>(
    (best, l) => (best == null || l.p > best.p ? l : best),
    undefined,
  );
  if (a.abstained) {
    return (
      <>
        <div className="pl-giq-imajev-answer-value">{DASH}</div>
        <AbstainedNote />
      </>
    );
  }
  return (
    <>
      <div className="pl-giq-imajev-answer-value">
        {a.score == null ? DASH : f.num(a.score, 2)}
        {max != null && <span className="pl-giq-imajev-score-max"> / {max}</span>}
      </div>
      {nearest?.description && <p className="hint">{nearest.description}</p>}
      <ProbRows
        rows={levels.map((l) => ({
          label: l.description ? `${l.level} · ${l.description}` : l.level,
          p: l.p,
        }))}
        winners={new Set(modal ? [modal.description ? `${modal.level} · ${modal.description}` : modal.level] : [])}
      />
      <Footnotes confidence={a.confidence} unknown={a.unknown_probability ?? 0} />
    </>
  );
}

function NoulBody({ a }: { a: DecideAnswer }) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const verdict = noulVerdict(a.noul);
  const yes = a.noul == null ? null : Math.min(1, Math.max(0, a.noul));
  if (a.abstained) {
    return (
      <>
        <div className="pl-giq-imajev-answer-value">{DASH}</div>
        <AbstainedNote />
      </>
    );
  }
  return (
    <>
      <div className="pl-giq-imajev-answer-value">{verdict ? t(`decide.${verdict}`) : DASH}</div>
      {yes != null && (
        <>
          <components.ProgressBar
            value={yes * 100}
            variant="accent"
            label={t("decide.noulMeter", {
              yes: f.pct(yes * 100),
              no: f.pct((1 - yes) * 100),
            })}
            valueText={f.pct(yes * 100)}
            height={8}
            className="pl-giq-imajev-meter"
          />
          <div className="pl-giq-imajev-scale" aria-hidden="true">
            <span>
              {t("decide.yes")} · {f.pct(yes * 100)}
            </span>
            <span>
              {t("decide.no")} · {f.pct((1 - yes) * 100)}
            </span>
          </div>
        </>
      )}
      <Footnotes confidence={a.confidence} unknown={a.unknown_probability ?? 0} />
    </>
  );
}

function MultiBody({ a }: { a: DecideAnswer }) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const labels = a.labels ?? [];
  const probs = sortedProbs(a);
  if (a.abstained) {
    return (
      <>
        <div className="pl-giq-imajev-answer-value">{DASH}</div>
        <AbstainedNote />
      </>
    );
  }
  return (
    <>
      <div className="pl-giq-imajev-answer-value pl-giq-imajev-multi-value">
        {labels.length ? labels.join(", ") : t("decide.noneSelected")}
      </div>
      {a.threshold != null && <p className="hint">{t("decide.threshold", { value: f.pct(a.threshold * 100) })}</p>}
      <ProbRows rows={probs.map(([label, p]) => ({ label, p }))} winners={new Set(labels)} />
      <Footnotes confidence={a.confidence} unknown={a.unknown_probability ?? 0} />
    </>
  );
}

/* A distribution as shared progress bars: each row stacks its label over a
   full-width bar, so the bars stay readable on a phone where the old
   three-column grid crushed them. The winner reads semibold on an accented
   bar; the rest stays muted. */
function ProbRows({ rows, winners }: { rows: { label: string; p: number }[]; winners: Set<string> }) {
  const f = useFormat();
  if (!rows.length) return null;
  return (
    <ul className="pl-giq-imajev-probs">
      {rows.map((r) => {
        const win = winners.has(r.label);
        return (
          <li key={r.label} className={win ? "pl-giq-imajev-winner" : undefined}>
            <div className="pl-giq-imajev-row-head">
              <span className="pl-giq-imajev-label" title={r.label}>
                {r.label}
              </span>
              <span className="pl-giq-imajev-pct">{f.pct(r.p * 100)}</span>
            </div>
            <components.ProgressBar
              value={r.p * 100}
              variant={win ? "accent" : "neutral"}
              label={r.label}
              valueText={f.pct(r.p * 100)}
              height={8}
            />
          </li>
        );
      })}
    </ul>
  );
}

/* Confidence and can't-tell share one footnote line, the way the other
   panels join their status lines — a caveat, not a paragraph each. */
function Footnotes({ confidence, unknown }: { confidence?: number | null; unknown?: number }) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const parts: string[] = [];
  if (confidence != null) parts.push(t("decide.confidence", { value: f.pct(confidence * 100) }));
  if (unknown != null && unknown > 0) parts.push(t("decide.unknown", { value: f.pct(unknown * 100) }));
  if (!parts.length) return null;
  return <p className="hint">{parts.join(" · ")}</p>;
}

function AbstainedNote() {
  const { t } = useTranslation(NS);
  return (
    <p className="warn pl-giq-imajev-abstain">
      <components.Icon as={WarningIcon} size={14} />
      <span>{t("decide.abstained")}</span>
    </p>
  );
}
