// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { Modality } from "../api/types";

/* The address contract between the sandbox and whatever links into it (the
   Recipes view's "Test in sandbox"): #/sandbox/<tab>?model=<name>. The hash
   rather than app state carries it, so the link also works as a bookmark or
   in a new tab. The app's route parser ignores everything after "?", so the
   tab still resolves. The sandbox only preselects the model — it never runs
   anything on arrival: for an image model that is minutes of GPU and an
   eviction of the resident set, and pressing Run there is what loads it. */

/** The sandbox's panels, in tab order. #/sandbox alone opens chat. */
export const SANDBOX_TABS = ["chat", "tools", "t2i", "edit", "vision", "asr", "tts", "voice"] as const;
export type SandboxTab = (typeof SANDBOX_TABS)[number];

export const DEFAULT_SANDBOX_TAB: SandboxTab = "chat";

export const isSandboxTab = (s: string | null | undefined): s is SandboxTab =>
  !!s && (SANDBOX_TABS as readonly string[]).includes(s);

/* Which panel exercises each worker, for a link that names no tab and for
   the models view's menu. Workers with no panel (OCR, depth) are
   absent: their "Test in sandbox" is offered disabled, with the reason. */
export const SANDBOX_TAB_FOR_WORKER: Partial<Record<Modality, SandboxTab>> = {
  llm: "chat",
  text2image: "t2i",
  image_edit: "edit",
  audio: "asr",
  stt: "asr",
  tts: "tts",
  embed: "voice",
};

export function sandboxHref(tab: SandboxTab, model: string): string {
  return `#/sandbox/${encodeURIComponent(tab)}?${new URLSearchParams({ model }).toString()}`;
}

export interface SandboxRoute {
  /** The tab named in the hash, or null when none (or an unknown one) is. */
  tab: SandboxTab | null;
  model: string | null;
}

/* Read tolerantly: the query may sit after the tab or straight after
   #/sandbox, may carry other keys, and a malformed escape or an unknown tab
   is just no tab (the caller falls back to chat). */
export function parseSandboxHash(hash: string): SandboxRoute {
  const body = hash.replace(/^#\/?/, "");
  const q = body.indexOf("?");
  const path = q < 0 ? body : body.slice(0, q);
  const parts = path.split("/").filter(Boolean);
  let tab: SandboxTab | null = null;
  if (parts[0] === "sandbox" && parts[1]) {
    let seg = parts[1];
    try {
      seg = decodeURIComponent(seg);
    } catch {
      /* a malformed escape is just not a tab */
    }
    tab = isSandboxTab(seg) ? seg : null;
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
