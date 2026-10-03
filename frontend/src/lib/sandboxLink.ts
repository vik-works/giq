// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { Modality } from "../api/types";
import { pluginPanels } from "../plugins/panels";

/* The address contract between the sandbox and whatever links into it (the
   Recipes view's "Test in sandbox"): #/sandbox/<tab>?model=<name>. The hash
   rather than app state carries it, so the link also works as a bookmark or
   in a new tab. The app's route parser ignores everything after "?", so the
   tab still resolves. The sandbox only preselects the model — it never runs
   anything on arrival: for an image model that is minutes of GPU and an
   eviction of the resident set, and pressing Run there is what loads it. */

/** The dashboard's own panels, for core's modality (llm), in tab order;
 * plugins' panels follow them (ADR-004 D6). #/sandbox alone opens chat. */
export const CORE_TABS = ["chat", "tools", "vision"] as const;
/** A panel's id: a core tab, or one a plugin's manifest declares. */
export type SandboxTab = string;

export const DEFAULT_SANDBOX_TAB: SandboxTab = "chat";

/* Which panel exercises a modality, for a link that names no tab and for
   the Recipes view's menu: chat for LLMs, else the first plugin panel
   declared for it. A modality with no panel (OCR, depth) has none, and its
   "Test in sandbox" is offered disabled, with the reason. */
export function tabForModality(modality: Modality | string): SandboxTab | null {
  if (modality === "llm") return "chat";
  return pluginPanels().find((p) => p.modality === modality)?.id ?? null;
}

export function sandboxHref(tab: SandboxTab, model: string): string {
  return `#/sandbox/${encodeURIComponent(tab)}?${new URLSearchParams({ model }).toString()}`;
}

export interface SandboxRoute {
  /** The tab named in the hash, or null when none (or an unknown one) is. */
  tab: SandboxTab | null;
  model: string | null;
}

/* Read tolerantly: the query may sit after the tab or straight after
   #/sandbox, may carry other keys, and a malformed escape is just no tab.
   A tab no panel has is the sandbox's to replace with chat. */
export function parseSandboxHash(hash: string): SandboxRoute {
  const body = hash.replace(/^#\/?/, "");
  const q = body.indexOf("?");
  const path = q < 0 ? body : body.slice(0, q);
  const parts = path.split("/").filter(Boolean);
  let tab: SandboxTab | null = null;
  if (parts[0] === "sandbox" && parts[1]) {
    try {
      // Whether a panel of that id exists is the sandbox's to say: plugins add them.
      tab = decodeURIComponent(parts[1]);
    } catch {
      /* a malformed escape is just not a tab */
    }
  }
  let model: string | null = null;
  if (q >= 0) {
    try {
      model = new URLSearchParams(body.slice(q + 1)).get("model")?.trim() || null;
    } catch {
      model = null;
    }
  }
  return { tab, model };
}
