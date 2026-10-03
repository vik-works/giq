// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType } from "react";
import type { ModelOption } from "../src/views/sandbox/models";

/* The contract between the dashboard and a plugin's UI (ADR-004 D6), types
   only: the dashboard and the plugins both import it. */

export type { ModelOption };

/** What the sandbox hands a plugin's panel. */
export interface PanelProps {
  /** The recipes serving the panel's modality that can load here, best first. */
  options: ModelOption[];
  /** The chosen one ("" when there is none). */
  model: string;
  onModel: (model: string) => void;
}

/** A panel as the manifest declares it: enough to draw its tab and link to
 * it without loading the plugin's code. */
export interface PanelMeta {
  /** The tab's id, its address: #/sandbox/<id>. */
  id: string;
  /** The modality it exercises: its icon, its recipes, "Test in sandbox". */
  modality: string;
  /** A WorkerIcon key; the modality's when absent. */
  icon?: string;
  /** Key of the tab's label in the plugin's namespace (plugin-<name>). */
  label: string;
}

/** ui/manifest.json. */
export interface UiManifest {
  api_version: number;
  /** The ES module, relative to the manifest. */
  module: string;
  styles?: string[];
  /** Language -> strings file; English is the fallback. */
  locales?: Record<string, string>;
  panels?: PanelMeta[];
}

/** A manifest as /capabilities lists it. */
export interface ServedUi extends UiManifest {
  plugin: string;
  /** URL prefix its files are served under. */
  base: string;
}

/** What a plugin's module exports: one component per panel id in its manifest. */
export interface PluginModule {
  panels: Record<string, ComponentType<PanelProps>>;
}
