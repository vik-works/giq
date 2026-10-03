// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { ArrowLeftIcon } from "@phosphor-icons/react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { errorText } from "../../api/client";
import { EmptyState } from "../../components/EmptyState";
import { Icon } from "../../components/Icon";
import { WorkerIcon } from "../../components/WorkerIcon";
import { hrefFor } from "../../lib/useHashRoute";
import { useDownloads, usePlugins, useRecipes } from "../../state";
import { addOffer, downloadOf, verdict } from "./adding";
import { AddRecipeRow } from "./AddRecipeRow";
import { PluginOffer } from "./PluginOffer";
import { useAddActions } from "./useAddActions";
import "./AddRecipes.css";
import { useModalityLabel } from "../../lib/modalities";

/* The Add page (ADR-005 D6): everything giq could run that is not on this
   machine, by the modality it adds, opened on purpose rather than pushed
   under the recipes that run. Recipes that cannot run here at all come
   last, folded, with the reason. */
export function AddRecipes() {
  const { t } = useTranslation("recipes");
  const modalityLabel = useModalityLabel();
  const recipes = useRecipes();
  const plugins = usePlugins();
  const downloads = useDownloads();
  const actions = useAddActions();
  const [query, setQuery] = useState("");
  const offer = useMemo(
    () => addOffer(recipes.data?.recipes ?? [], plugins.data?.plugins ?? [], query.trim()),
    [recipes.data, plugins.data, query],
  );
  const list = downloads.data?.downloads;
  const nothing = offer.groups.length === 0 && offer.enginePlugins.length === 0 && offer.unfit.length === 0;

  return (
    <div className="rc-add">
      <div className="rc-add-bar">
        <a className="btn btn-ghost btn-sm" href={hrefFor("recipes")}>
          <Icon as={ArrowLeftIcon} size={14} />
          {t("add.back")}
        </a>
        <input
          type="search"
          className="input rc-add-search"
          placeholder={t("add.search")}
          aria-label={t("add.search")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <p className="hint rc-add-intro">{t("add.intro")}</p>
      {!recipes.data ? (
        <EmptyState>
          {recipes.error ? t("common:empty.failed", { error: errorText(recipes.error) }) : t("common:empty.loading")}
        </EmptyState>
      ) : nothing ? (
        <EmptyState>{query ? t("add.noMatch") : t("add.nothing")}</EmptyState>
      ) : null}
      {offer.groups.map((g) => (
        <section key={g.modality} className="rc-add-group" aria-labelledby={`rc-add-${g.modality}`}>
          <h2 id={`rc-add-${g.modality}`} className="section-title rc-add-group-title">
            <WorkerIcon worker={g.modality} size={14} />
            {modalityLabel(g.modality)}
          </h2>
          <div className="rc-add-grid">
            {g.offered.map((r) => (
              <AddRecipeRow key={r.name} r={r} download={downloadOf(r.name, list)} actions={actions} />
            ))}
            {g.plugins.map((p) => (
              <PluginOffer key={p.name} p={p} />
            ))}
          </div>
        </section>
      ))}
      {offer.enginePlugins.length > 0 && (
        <section className="rc-add-group" aria-labelledby="rc-add-engines">
          <h2 id="rc-add-engines" className="section-title rc-add-group-title">
            {t("add.engines")}
          </h2>
          <div className="rc-add-grid">
            {offer.enginePlugins.map((p) => (
              <PluginOffer key={p.name} p={p} />
            ))}
          </div>
        </section>
      )}
      {offer.unfit.length > 0 && (
        <details className="rc-add-unfit">
          <summary>{t("add.unfit", { count: offer.unfit.length })}</summary>
          <ul>
            {offer.unfit.map((r) => (
              <li key={r.name}>
                <span className="mono">{r.name}</span> — {verdict(r)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
