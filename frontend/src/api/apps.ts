// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { del, postJSON } from "./client";
import type { AppCreated, AppEntry } from "./types";

/* The writes on apps (ADR-006 D6). The list itself is a shared poller
   (state/useApps), so the Usage view's filter and the Apps page agree. */

const path = (name: string) => `/apps/${encodeURIComponent(name)}`;

/** A new app and its key: the key is in this response and nowhere else. */
export const createApp = (name: string, description?: string): Promise<AppCreated> =>
  postJSON<AppCreated>("/apps", description ? { name, description } : { name });

/** Stop the key working; the history stays and the name stays taken. */
export const revokeApp = (name: string): Promise<AppEntry> => postJSON<AppEntry>(`${path(name)}/revoke`);

/** Remove the entry; its jobs stay in the stats under its name. */
export const deleteApp = (name: string): Promise<void> => del<void>(path(name));
