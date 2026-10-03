// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type * as I18nNS from "react-i18next";

/* "react-i18next" for a plugin's build: the dashboard's, so a plugin's
   useTranslation sees the strings the dashboard loaded for it. */
const I = (globalThis.__GIQ_HOST__ as { reactI18next: typeof I18nNS }).reactI18next;

export const { Trans, useTranslation } = I;
