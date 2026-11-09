// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import { components } from "@giq/plugin-ui";
import type { ImageOptionsValue } from "./imageTask";
import { NS } from "./ns";
import "./ImageOptions.css";

const { Field } = components;

interface Props {
  value: ImageOptionsValue;
  onChange: (next: ImageOptionsValue) => void;
  /** What an empty size stands for: the configured default, or the reference's size. */
  sizeHint: string;
}

/* Seed, size and negative prompt: the optional task fields both image panels share. */
export function ImageOptions({ value, onChange, sizeHint }: Props) {
  const { t } = useTranslation(NS);
  const set = (key: keyof ImageOptionsValue) => (e: React.ChangeEvent<HTMLInputElement>) =>
    onChange({ ...value, [key]: e.target.value });
  const number = (key: "seed" | "width" | "height", label: string, placeholder: string) => (
    <Field label={label} className="pl-giq-sdcpp-num">
      {(id) => (
        <input id={id} className="input" type="number" min={key === "seed" ? 0 : 1} placeholder={placeholder} value={value[key]} onChange={set(key)} />
      )}
    </Field>
  );
  return (
    <>
      <div className="sbx-row">
        {number("width", t("field.width"), t("field.sizeDefault"))}
        {number("height", t("field.height"), t("field.sizeDefault"))}
        {number("seed", t("field.seed"), t("field.seedRandom"))}
      </div>
      <p className="hint">{sizeHint}</p>
      <Field label={t("field.negative")}>
        {(id) => <input id={id} className="input" type="text" value={value.negative} onChange={set("negative")} />}
      </Field>
    </>
  );
}
