// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useSyncExternalStore } from "react";

/* Hash routes: #/, #/recipes, #/inventory, #/usage, #/sandbox.
   #/sandbox/<tab> additionally deep-links a sandbox tab, and #/recipes/add
   is the page that adds recipes to this machine (ADR-005). #/models — the
   Recipes view's address before ADR-003 — still lands there, so bookmarks
   keep working. Anything else is the overview. No router library: five
   views do not need one. */

export const VIEWS = ["overview", "recipes", "inventory", "usage", "sandbox"] as const;
export type View = (typeof VIEWS)[number];

export interface Route {
  view: View;
  /** The segment after the view, e.g. the sandbox tab in #/sandbox/chat. */
  sub: string | null;
}

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split(/[?#]/)[0]!.split("/").filter(Boolean);
  const head = parts[0];
  if (head === "usage" || head === "inventory") return { view: head, sub: null };
  if (head === "recipes") return { view: "recipes", sub: parts[1] === "add" ? "add" : null };
  if (head === "models") return { view: "recipes", sub: null };
  if (head === "sandbox") return { view: "sandbox", sub: parts[1] ? decodeURIComponent(parts[1]) : null };
  return { view: "overview", sub: null };
}

/** The hash for a route: hrefFor("sandbox", "chat") → "#/sandbox/chat". */
export function hrefFor(view: View, sub?: string | null): string {
  if (view === "overview") return "#/";
  return `#/${view}${sub ? "/" + encodeURIComponent(sub) : ""}`;
}

function subscribe(cb: () => void): () => void {
  window.addEventListener("hashchange", cb);
  return () => window.removeEventListener("hashchange", cb);
}
const snapshot = () => window.location.hash;

export function useHashRoute(): Route & {
  navigate: (view: View, sub?: string | null, opts?: { replace?: boolean }) => void;
} {
  const hash = useSyncExternalStore(subscribe, snapshot, () => "");
  const route = parseHash(hash);
  const navigate = useCallback(
    (view: View, sub?: string | null, opts?: { replace?: boolean }) => {
      const next = hrefFor(view, sub);
      if (opts?.replace) {
        // replaceState fires no hashchange; dispatch one so subscribers see it.
        history.replaceState(null, "", next);
        window.dispatchEvent(new HashChangeEvent("hashchange"));
      } else {
        window.location.hash = next;
      }
    },
    [],
  );
  return { ...route, navigate };
}
