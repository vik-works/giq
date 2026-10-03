// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import { PlusIcon } from "@phosphor-icons/react";
import { EmptyState } from "../../components/EmptyState";
import { Icon } from "../../components/Icon";
import { hrefFor } from "../../lib/useHashRoute";
import { ActionStatus } from "./ActionStatus";
import type { CardChoice } from "../../lib/cards";
import type { RecipeEntry, WeightsItem } from "../../api/types";
import { RecipeCard } from "./RecipeCard";
import type { RecipeActions } from "./useRecipeActions";
import "./RecipeCatalog.css";

export interface RecipeCatalogProps {
  /** The entries passing the filters. */
  entries: RecipeEntry[];
  weights: WeightsItem[] | undefined;
  total: number;
  /** Recipes giq knows that are not on this machine: the Add page's. */
  addable: number;
  filtered: boolean;
  onClearFilters: () => void;
  cards: CardChoice[];
  versions: Map<string, string | null>;
  actions: RecipeActions;
}

/* The catalog: a card for each recipe on this machine. The ones giq knows
   but has no weights for used to follow as a line of pills; they are not
   what runs here, so they live on the Add page, one link away (ADR-005). */
export function RecipeCatalog({ entries, weights, total, addable, filtered, onClearFilters, cards, versions, actions }: RecipeCatalogProps) {
  const { t } = useTranslation("recipes");
  return (
    <section className="rc-recipe-catalog" aria-labelledby="rc-recipe-catalog-title">
      <div className="rc-catalog-bar">
        <h2 id="rc-recipe-catalog-title" className="section-title rc-catalog-title">
          {t("catalog.title")}
        </h2>
        <span className="rc-catalog-count">
          {filtered
            ? t("count.filtered", { count: entries.length, total })
            : t("count.all", { count: entries.length })}
        </span>
        <a className="btn btn-secondary btn-sm rc-catalog-add" href={hrefFor("recipes", "add")}>
          <Icon as={PlusIcon} size={14} />
          {addable > 0 ? t("catalog.add", { count: addable }) : t("catalog.addNone")}
        </a>
      </div>
      <ActionStatus message={actions.message} onDismiss={actions.dismiss} />
      {entries.length === 0 ? (
        <EmptyState
          action={
            filtered && (
              <button type="button" className="btn btn-secondary" onClick={onClearFilters}>
                {t("catalog.clearFilters")}
              </button>
            )
          }
        >
          {total === 0 && !filtered ? t("catalog.nothingHere") : t("catalog.noMatch")}
        </EmptyState>
      ) : (
        <div className="rc-recipe-grid">
          {entries.map((r) => (
            <RecipeCard key={r.name} r={r} weights={weights} cards={cards} versions={versions} actions={actions} />
          ))}
        </div>
      )}
    </section>
  );
}
