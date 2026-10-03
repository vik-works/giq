// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { WorkerIcon } from "./WorkerIcon";
import "./ModelLabel.css";
import { useModalityLabel } from "../lib/modalities";

export interface ModelLabelProps {
  worker: string;
  model: string;
  /** The model's series colour; a swatch ties the row to a chart. Omit for no swatch. */
  color?: string;
  /** Show the worker's name after the model (the same model can exist under two workers). */
  showWorker?: boolean;
}

/** A model as the job and usage tables name it: swatch, worker icon, name, worker. */
export function ModelLabel({ worker, model, color, showWorker }: ModelLabelProps) {
  const modalityLabel = useModalityLabel();
  return (
    <span className="model-label">
      {color && <span className="model-label-swatch" style={{ background: color }} aria-hidden />}
      <WorkerIcon worker={worker} size={13} labelled={!showWorker} />
      <span className="model-label-name">{model}</span>
      {showWorker && (
        <span className="model-label-worker">· {modalityLabel(worker)}</span>
      )}
    </span>
  );
}
