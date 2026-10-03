// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type * as JsxNS from "react/jsx-runtime";

/* "react/jsx-runtime" for a plugin's build: the dashboard's. */
const J = (globalThis.__GIQ_HOST__ as { jsxRuntime: typeof JsxNS }).jsxRuntime;

export const { Fragment, jsx, jsxs } = J;
