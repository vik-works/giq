// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { AppEntry } from "../../api/types";

/* The Apps page's arithmetic, free of React: order, status, the server's
   ISO dates as the epoch seconds lib/format takes, and the snippets that
   show where a key goes. */

export type AppStatus = "active" | "revoked";

export const appStatus = (a: AppEntry): AppStatus => (a.revoked ? "revoked" : "active");

/** An ISO 8601 timestamp as epoch seconds, or null when absent or unreadable. */
export function isoSeconds(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : ms / 1000;
}

/** Active apps first, then by name: a revoked entry is history, kept only to hold its name. */
export const sortApps = (apps: AppEntry[]): AppEntry[] =>
  [...apps].sort((a, b) => Number(!!a.revoked) - Number(!!b.revoked) || a.name.localeCompare(b.name));

/** The Usage view filtered to one app. */
export const usageHref = (name: string): string => `#/usage?app=${encodeURIComponent(name)}`;

/* Where the key goes: the OpenAI SDKs' api_key, which they send as
   Authorization: Bearer. The base URL is this dashboard's own origin, as
   that is the address the operator reached giq by. */
export function keySnippets(origin: string, key: string): { python: string; curl: string } {
  const base = `${origin.replace(/\/+$/, "")}/v1`;
  return {
    python: `from openai import OpenAI\n\nclient = OpenAI(base_url="${base}", api_key="${key}")`,
    curl: `curl ${base}/models -H "Authorization: Bearer ${key}"`,
  };
}
