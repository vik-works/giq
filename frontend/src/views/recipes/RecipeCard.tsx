// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { EyeIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import { Icon } from "../../components/Icon";
import { Tag } from "../../components/Tag";
import { WorkerIcon } from "../../components/WorkerIcon";
import { DeviceSelect } from "./DeviceSelect";
import { EngineName } from "./EngineName";
import type { CardChoice } from "../../lib/cards";
import type { RecipeEntry, WeightsItem } from "../../api/types";
import { primaryModality } from "../../lib/recipes";
import { verdict } from "./adding";
import { FIT_TONE, weightsOf } from "./catalog";
import { RecipeCardMenu } from "./RecipeCardMenu";
import { RecipeFacts } from "./RecipeFacts";
import { PolicyControl } from "./PolicyControl";
import type { RecipeActions } from "./useRecipeActions";
import "./RecipeCard.css";

export interface RecipeCardProps {
  r: RecipeEntry;
  weights: WeightsItem[] | undefined;
  cards: CardChoice[];
  versions: Map<string, string | null>;
  actions: RecipeActions;
}

/* One recipe, one card. A row per recipe in a wide table made every recipe
   look like a spreadsheet line; the unit of work on this page is one recipe:
   what it is, whether it fits, how it is kept, where it runs, what runs it. */
export function RecipeCard({ r, weights, cards, versions, actions }: RecipeCardProps) {
  const { t } = useTranslation("recipes");
  const busy = actions.isBusy(r);
  const modality = primaryModality(r);
  const policy = r.residency.policy;
  return (
    <article className={`card elev-sm rc-recipe-card${policy === "off" ? " rc-recipe-off" : ""}`} aria-busy={busy}>
      <header className="rc-recipe-head">
        <WorkerIcon worker={modality} size={16} labelled />
        <h3 className="rc-recipe-name">{r.name}</h3>
        <span className="rc-recipe-badges">
          {policy === "pinned" && (
            <Tag tone="accent" title={t("card.keptWarmTitle")}>
              {t("card.keptWarm")}
            </Tag>
          )}
          {/* A capability that changes what you can send it; same class of
              fact as the engine row and the tilde on estimated VRAM. */}
          {r.vision && (
            <Tag tone="outline" icon={<Icon as={EyeIcon} size={12} />} title={t("card.visionTitle")}>
              {t("card.vision")}
            </Tag>
          )}
          {/* On disk but not runnable here: the engine binary is missing, or no
              card can take it. The reason is the server's, as the title. */}
          {r.availability === "unfit" ? (
            <Tag tone="critical" title={verdict(r) ?? undefined}>
              {t("add.availability.unfit")}
            </Tag>
          ) : (
            <Tag tone={FIT_TONE[r.fit] ?? "neutral"}>{t(`common:fit.${r.fit}`, { defaultValue: r.fit })}</Tag>
          )}
        </span>
        <RecipeCardMenu r={r} />
      </header>
      <p className="rc-recipe-meta">
        {r.modalities.map((x) => t(`common:worker.${x}`, { defaultValue: x })).join(" · ")}
        {r.detail && <> · {r.detail}</>}
      </p>
      <RecipeFacts r={r} weights={weightsOf(r, weights)} />
      <dl className="rc-recipe-rows">
        <div className="rc-recipe-row">
          <dt>{t("card.residency")}</dt>
          <dd>
            <PolicyControl
              r={r}
              busy={busy}
              onChange={(p) => void actions.changePolicy(r, p)}
              onRevert={() => void actions.revert(r)}
            />
          </dd>
        </div>
        <div className="rc-recipe-row">
          <dt>{t("card.gpu")}</dt>
          <dd>
            <DeviceSelect r={r} cards={cards} busy={busy} onBind={(d, name) => void actions.bind(r, d, name)} />
          </dd>
        </div>
        <div className="rc-recipe-row">
          <dt>{t("card.engine")}</dt>
          <dd>
            <EngineName r={r} versions={versions} />
          </dd>
        </div>
      </dl>
    </article>
  );
}
