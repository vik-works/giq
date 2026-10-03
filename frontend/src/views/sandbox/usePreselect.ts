// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import type { RecipesResponse } from "../../api/types";
import { navigateReplace } from "./navigate";
import { tabOptions, type SandboxModels } from "./models";
import type { SandboxRoute } from "../../lib/sandboxLink";
import { DEFAULT_TAB, tabForModality, type TabInfo } from "./tabs";

/* "Test in sandbox" arrives as #/sandbox/<tab>?model=<name>. Once the
   recipes are known the model is chosen in that panel and the query leaves
   the address, so the link is consumed once and a later recipes refresh
   cannot re-apply it over a choice made by hand. Nothing runs: pressing Run
   is what loads the model. A link without a tab goes to the panel for the
   model's modality. Returns the model (and its panel) when the panel cannot
   take it — never-fits, or no such model — so the view can say so. */
export interface PreselectMiss {
  tab: string;
  model: string;
}

export function usePreselect(
  route: SandboxRoute,
  recipes: RecipesResponse | undefined,
  models: SandboxModels,
  tabs: TabInfo[],
  choose: (tab: string, model: string) => void,
): PreselectMiss | null {
  const [missed, setMissed] = useState<PreselectMiss | null>(null);
  const { model } = route;

  useEffect(() => {
    if (!model || !recipes) return;
    const modality = recipes.recipes.find((r) => r.name === model || r.aliases.includes(model))?.modalities[0];
    const id = route.tab ?? (modality && tabForModality(modality)) ?? DEFAULT_TAB;
    const tab = tabs.find((t) => t.id === id);
    const options = tab ? tabOptions(models, tab) : [];
    if (tab && options.length) {
      const ok = options.some((o) => o.model === model);
      if (ok) choose(tab.id, model);
      setMissed(ok ? null : { tab: tab.id, model });
    } else {
      // A panel without a model select opens, and runs what its lane has.
      setMissed(null);
    }
    navigateReplace(id);
  }, [model, route.tab, recipes, models, tabs, choose]);

  return missed;
}
