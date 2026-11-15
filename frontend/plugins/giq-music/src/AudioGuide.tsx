// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import { components } from "@giq/plugin-ui";
import { AUDIO_USES, TRACKS, type AudioGuideValue } from "./audioTask";
import { NS } from "./ns";

const { Field, FilePicker, SegmentedControl } = components;

interface Props {
  value: AudioGuideValue;
  onChange: (next: AudioGuideValue) => void;
}

/** Audio of the user's own for an ACE-Step track: a sound to match, or a source to build on. */
export function AudioGuide({ value, onChange }: Props) {
  const { t } = useTranslation(NS);
  const set = (patch: Partial<AudioGuideValue>) => onChange({ ...value, ...patch });
  return (
    <>
      <Field label={t("field.ownAudio")} hint={value.file ? t(`audio.${value.use}Hint`) : t("audio.hint")}>
        {(id) => <FilePicker id={id} accept="audio/*" onChange={(file) => set({ file })} />}
      </Field>
      {value.file && (
        <div className="sbx-row">
          <div className="field form-field">
            <span className="form-label" aria-hidden>{t("field.audioUse")}</span>
            <SegmentedControl
              label={t("field.audioUse")}
              value={value.use}
              options={AUDIO_USES.map((use) => ({ value: use, label: t(`audio.${use}`) }))}
              onChange={(use) => set({ use: use as AudioGuideValue["use"] })}
            />
          </div>
          {value.use === "cover" && (
            <Field label={t("field.coverStrength")} className="sbx-narrow">
              {(id) => (
                <input
                  id={id}
                  className="input"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  placeholder="0.5"
                  value={value.strength}
                  onChange={(e) => set({ strength: e.target.value })}
                />
              )}
            </Field>
          )}
          {value.use === "lego" && (
            <Field label={t("field.track")}>
              {(id) => (
                <select
                  id={id}
                  className="input"
                  value={value.track}
                  onChange={(e) => set({ track: e.target.value as AudioGuideValue["track"] })}
                >
                  {TRACKS.map((track) => (
                    <option key={track} value={track}>
                      {t(`track.${track}`)}
                    </option>
                  ))}
                </select>
              )}
            </Field>
          )}
        </div>
      )}
    </>
  );
}
