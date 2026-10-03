// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import { components, useFormat } from "@giq/plugin-ui";
import type { TagTone } from "../../../src/components/Tag";
import { verdict, type Verdict } from "./cosine";
import { NS } from "./ns";
import "./CosineResult.css";

const { ProgressBar, Tag } = components;

const TONE: Record<Verdict, TagTone> = { same: "good", inconclusive: "warning", different: "neutral" };

/** The similarity as a number, a gauge and a plain-language verdict. */
export function CosineResult({ cos }: { cos: number }) {
  const { t } = useTranslation(NS);
  const f = useFormat();
  const v = verdict(cos);
  const pct = Math.max(0, Math.min(100, cos * 100));
  return (
    <div className="pl-giq-speech-cos">
      <div className="pl-giq-speech-cos-head">
        <span className="pl-giq-speech-cos-value">{t("voice.cosine", { value: f.num(cos, 3) })}</span>
        <Tag tone={TONE[v]}>{t(`voice.tag.${v}`)}</Tag>
      </div>
      <ProgressBar value={pct} label={t("voice.gauge")} valueText={f.num(cos, 3)} height={8} />
      <p className="pl-giq-speech-verdict">{t(`voice.verdict.${v}`)}</p>
    </div>
  );
}
