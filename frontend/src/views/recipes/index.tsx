// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { errorText } from "../../api/client";
import { EmptyState } from "../../components/EmptyState";
import { PageHeader } from "../../components/PageHeader";
import { WorkerIcon } from "../../components/WorkerIcon";
import { useEngines, useGpus, useRecipes, useStorage, useWeights } from "../../state";
import { cardChoices } from "../../lib/cards";
import { useHashRoute } from "../../lib/useHashRoute";
import { AddRecipes } from "./AddRecipes";
import { onThisMachine } from "./adding";
import { ENGINE_NOTE_KEY, engineVersions, facetCounts, matches, toggled } from "./catalog";
import { FacetFilter } from "./FacetFilter";
import { RecipeErrors } from "./RecipeErrors";
import { RecipeCatalog } from "./RecipeCatalog";
import { ResidencyBudgets } from "./ResidencyBudgets";
import { useRecipeActions } from "./useRecipeActions";
import "./RecipesView.css";

const EMPTY: ReadonlySet<string> = new Set();

/* The Recipes page: filters on the left, the recipes on this machine in the
   middle, what is kept warm per card on the right. Residency used to be a
   full-width section above the catalog, which pushed the recipes — the
   reason to open the page — below the fold. On a narrow screen the filters
   stay on top and the context moves under the catalog. What is on disk is
   the Inventory's; what could be added is #/recipes/add (ADR-005). */
export default function RecipesView() {
  const { sub } = useHashRoute();
  const { t } = useTranslation("recipes");
  if (sub === "add") {
    return (
      <>
        <PageHeader title={t("add.title")} />
        <div className="rc-view">
          <AddRecipes />
        </div>
      </>
    );
  }
  return <OnThisMachine />;
}

function OnThisMachine() {
  const { t } = useTranslation("recipes");
  const recipes = useRecipes();
  const weights = useWeights();
  const storage = useStorage();
  const gpus = useGpus();
  const engines = useEngines();
  const actions = useRecipeActions();
  const [modality, setModality] = useState<ReadonlySet<string>>(EMPTY);
  const [engine, setEngine] = useState<ReadonlySet<string>>(EMPTY);

  const known = recipes.data?.recipes ?? [];
  const all = known.filter(onThisMachine);
  const shown = all.filter((r) => matches(r, { modality, engine }));
  const cards = useMemo(() => cardChoices(gpus.data?.gpus, recipes.data?.cards), [gpus.data, recipes.data]);
  const versions = useMemo(() => engineVersions(engines.data), [engines.data]);
  const noteFor = (engineName: string) =>
    ENGINE_NOTE_KEY[engineName] ? t(`engineNote.${ENGINE_NOTE_KEY[engineName]}`) : undefined;
  const clear = () => {
    setModality(EMPTY);
    setEngine(EMPTY);
  };

  return (
    <>
      <PageHeader title={t("common:nav.recipes")} />
      <div className="rc-view">
        <div className="rc-layout">
          <aside className="rc-filters" aria-label={t("sidebar.label")}>
            <FacetFilter
              title={t("sidebar.modality")}
              items={facetCounts(all, (r) => r.modalities)}
              selected={modality}
              onToggle={(v) => setModality((s) => toggled(s, v))}
              onClear={() => setModality(EMPTY)}
              renderLabel={(w) => (
                <>
                  <WorkerIcon worker={w} size={14} />
                  {t(`common:worker.${w}`, { defaultValue: w })}
                </>
              )}
            />
            <FacetFilter
              title={t("sidebar.engine")}
              items={facetCounts(all, (r) => [r.engine])}
              selected={engine}
              onToggle={(v) => setEngine((s) => toggled(s, v))}
              onClear={() => setEngine(EMPTY)}
              renderLabel={(e) => <span className="mono">{e}</span>}
              titleFor={noteFor}
            />
          </aside>
          <div className="rc-main">
            <RecipeErrors recipes={storage.data?.recipes} />
            {recipes.data ? (
              <RecipeCatalog
                entries={shown}
                weights={weights.data?.weights}
                total={all.length}
                addable={known.length - all.length}
                filtered={modality.size > 0 || engine.size > 0}
                onClearFilters={clear}
                cards={cards}
                versions={versions}
                actions={actions}
              />
            ) : (
              <EmptyState>
                {recipes.error
                  ? t("common:empty.failed", { error: errorText(recipes.error) })
                  : t("common:empty.loading")}
              </EmptyState>
            )}
          </div>
          <aside className="rc-context" aria-label={t("sidebar.residency")}>
            <ResidencyBudgets recipes={recipes.data} />
          </aside>
        </div>
      </div>
    </>
  );
}
