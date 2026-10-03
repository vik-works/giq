// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useSyncExternalStore } from "react";
import type { PanelMeta, ServedUi } from "../../plugin-ui/types";

/* The sandbox panels plugins declare, from their manifests in /capabilities:
   enough to draw the tabs and to link a recipe to its panel without loading
   any plugin's code. Filled by the data layer, like the modality registry. */

export interface PluginPanel extends PanelMeta {
  plugin: string;
  ui: ServedUi;
}

let panels: readonly PluginPanel[] = [];
let uis: readonly ServedUi[] = [];
const listeners = new Set<() => void>();

export function setPluginUis(next: readonly ServedUi[]): void {
  if (JSON.stringify(next) === JSON.stringify(uis)) return;
  uis = next;
  panels = next.flatMap((ui) => (ui.panels ?? []).map((p) => ({ ...p, plugin: ui.plugin, ui })));
  for (const l of listeners) l();
}

const subscribe = (cb: () => void) => {
  listeners.add(cb);
  return () => listeners.delete(cb);
};

export const pluginPanels = (): readonly PluginPanel[] => panels;
export const pluginUis = (): readonly ServedUi[] => uis;

export function usePluginPanels(): readonly PluginPanel[] {
  return useSyncExternalStore(subscribe, pluginPanels, pluginPanels);
}
