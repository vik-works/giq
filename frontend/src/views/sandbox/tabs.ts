// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { PluginPanel } from "../../plugins/panels";
import { CORE_TABS, type SandboxTab as Tab } from "../../lib/sandboxLink";

/* The tab list: the dashboard's own panels for LLMs, then each plugin's, in
   the server's registration order (ADR-004 D6). Ids are the sandbox's
   address contract (lib/sandboxLink.ts); a plugin keeps an id once it ships
   one, or bookmarks break. */
export {
  DEFAULT_SANDBOX_TAB as DEFAULT_TAB,
  tabForModality,
  type SandboxTab as Tab,
} from "../../lib/sandboxLink";

export type CoreTab = (typeof CORE_TABS)[number];

export interface TabInfo {
  id: Tab;
  /** A WorkerIcon key. */
  icon: string;
  /** The modality whose recipes it offers. */
  modality: string;
  /** Set for a plugin's panel. */
  plugin?: PluginPanel;
}

const CORE: Record<CoreTab, TabInfo> = {
  chat: { id: "chat", icon: "llm", modality: "llm" },
  tools: { id: "tools", icon: "llm", modality: "llm" },
  vision: { id: "vision", icon: "vision", modality: "llm" },
};

export const isCoreTab = (t: Tab): t is CoreTab => (CORE_TABS as readonly string[]).includes(t);

export function sandboxTabs(panels: readonly PluginPanel[]): TabInfo[] {
  const seen = new Set<string>(CORE_TABS);
  const plugin = panels
    .filter((p) => !seen.has(p.id) && (seen.add(p.id), true))
    .map((p) => ({ id: p.id, icon: p.icon ?? p.modality, modality: p.modality, plugin: p }));
  return [...CORE_TABS.map((t) => CORE[t]), ...plugin];
}
