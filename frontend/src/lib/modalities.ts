// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useSyncExternalStore } from "react";
import { useTranslation } from "react-i18next";

/* The modalities as the server registers them (ADR-004 D6), in registration
   order, with the label and icon name each plugin gave. The dashboard's own
   strings and icons outrank these for the modalities it knows; a modality a
   plugin adds is drawn with what the server sends. Filled from
   /capabilities by the data layer; a module-level store, so the plain
   helpers (series colours) and the components read the same list. */

export interface ModalityMeta {
  name: string;
  label: string;
  icon: string;
}

let registered: readonly ModalityMeta[] = [];
const listeners = new Set<() => void>();

export function setModalities(next: readonly ModalityMeta[]): void {
  const same =
    next.length === registered.length &&
    next.every((m, i) => m.name === registered[i]!.name && m.label === registered[i]!.label && m.icon === registered[i]!.icon);
  if (same) return;
  registered = next;
  for (const l of listeners) l();
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export const modalities = (): readonly ModalityMeta[] => registered;

/** The registered modalities, re-rendering when the server's list changes. */
export function useModalities(): readonly ModalityMeta[] {
  return useSyncExternalStore(subscribe, modalities, modalities);
}

export const modalityMeta = (name: string): ModalityMeta | undefined =>
  registered.find((m) => m.name === name);

/** Position in registration order, or -1 for a modality the server does not list. */
export const registrationIndex = (name: string): number => registered.findIndex((m) => m.name === name);

/** A modality's display name: the dashboard's string, else the server's label, else the name. */
export function useModalityLabel(): (name: string) => string {
  const { t } = useTranslation();
  const list = useModalities();
  return useCallback(
    (name: string) =>
      t(`common:worker.${name}`, { defaultValue: list.find((m) => m.name === name)?.label || name }),
    [t, list],
  );
}
