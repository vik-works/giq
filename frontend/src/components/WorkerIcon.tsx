// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import {
  ChatTextIcon,
  EyeIcon,
  FileTextIcon,
  GraphicsCardIcon,
  ImageIcon,
  MountainsIcon,
  PaintBrushIcon,
  SpeakerHighIcon,
  VectorThreeIcon,
  WaveformIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import { workerColor } from "../lib/series";
import { Icon } from "./Icon";

/* Shape says what the model does; colour (optional) says which worker, in
   the same slot colour the charts use for it. Vision is a capability of an
   LLM rather than a worker, so it is its own key. */
export const WORKER_ICONS: Record<string, PhosphorIcon> = {
  llm: ChatTextIcon,
  vision: EyeIcon,
  text2image: ImageIcon,
  image_edit: PaintBrushIcon,
  audio: WaveformIcon,
  stt: WaveformIcon,
  tts: SpeakerHighIcon,
  embed: VectorThreeIcon,
  ocr: FileTextIcon,
  depth: MountainsIcon,
  gpu: GraphicsCardIcon,
};

export interface WorkerIconProps {
  /** A worker type, or "vision" / "gpu". */
  worker: string;
  size?: number;
  /** Tint with the worker's series colour (default) or inherit the text colour. */
  colored?: boolean;
  /** Announce the worker's name; otherwise decorative. */
  labelled?: boolean;
  className?: string;
}

export function WorkerIcon({ worker, size = 14, colored = true, labelled = false, className }: WorkerIconProps) {
  const { t } = useTranslation();
  const glyph = WORKER_ICONS[worker] ?? ChatTextIcon;
  return (
    <Icon
      as={glyph}
      size={size}
      className={className}
      style={{ flex: "none", verticalAlign: "-2px", color: colored ? workerColor(worker) : undefined }}
      label={labelled ? t(`worker.${worker}`, { defaultValue: worker }) : undefined}
    />
  );
}
