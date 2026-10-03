// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { GiqHost } from "../src/plugins/host";

/* @giq/plugin-ui at run time: what the dashboard lends a plugin's module.
   Import it like a library; the plugin's build leaves it to the host.

     import { api, components, sandbox, useFormat } from "@giq/plugin-ui";
*/

const host = globalThis.__GIQ_HOST__ as GiqHost;

export const API_VERSION = host.version;
export const api = host.api;
export const components = host.components;
export const sandbox = host.sandbox;
export const useFormat = host.useFormat;
export const useRunner = host.useRunner;

export type { GiqHost };
export type * from "./types";
