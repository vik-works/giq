// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import {
  ChatTextIcon,
  CubeIcon,
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
import { workerColor } from "../lib/series";
import { Icon } from "./Icon";
import { modalityMeta, useModalityLabel } from "../lib/modalities";

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

/* The icon names a plugin may give its modality (giq.plugin.Modality.icon),
   for modalities the table above does not know. Anything else gets the
   generic cube, never a wrong picture. */
const ICON_BY_NAME: Record<string, PhosphorIcon> = {
  chat: ChatTextIcon,
  eye: EyeIcon,
  image: ImageIcon,
  "paint-brush": PaintBrushIcon,
  waveform: WaveformIcon,
  speaker: SpeakerHighIcon,
  vector: VectorThreeIcon,
  "file-text": FileTextIcon,
  mountains: MountainsIcon,
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
  const modalityLabel = useModalityLabel();
  const glyph = WORKER_ICONS[worker] ?? ICON_BY_NAME[modalityMeta(worker)?.icon ?? ""] ?? CubeIcon;
  return (
    <Icon
      as={glyph}
      size={size}
      className={className}
      style={{ flex: "none", verticalAlign: "-2px", color: colored ? workerColor(worker) : undefined }}
      label={labelled ? modalityLabel(worker) : undefined}
    />
  );
}
