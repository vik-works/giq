// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { TFunction } from "i18next";
import { ANONYMOUS } from "./usageModel";

/* The two buckets that are not apps read as words in the reader's language;
   an app's own name is the operator's and stays as typed. */
export function appLabel(app: string | null, t: TFunction): string {
  if (app == null) return t("usage:app.unknown");
  if (app === ANONYMOUS) return t("usage:app.anonymous");
  return app;
}
