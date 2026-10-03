// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { PuzzlePieceIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import type { PluginEntry } from "../../api/types";
import { Icon } from "../../components/Icon";
import { CopyCommand } from "./CopyCommand";
import "./PluginOffer.css";

/* A curated plugin this giq does not have: what it adds, what it needs
   besides Python, and the command that installs it. The dashboard does not
   install it: a plugin changes giq's own environment and needs a restart,
   which is the operator's to do (ADR-005 D5). */
export function PluginOffer({ p }: { p: PluginEntry }) {
  const { t } = useTranslation("recipes");
  return (
    <article className="card rc-add-plugin">
      <header className="rc-add-plugin-head">
        <Icon as={PuzzlePieceIcon} size={16} />
        <h3 className="rc-add-plugin-name mono">{p.name}</h3>
      </header>
      <p className="rc-add-plugin-summary">{p.summary}</p>
      {p.recipes.length > 0 && (
        <p className="rc-add-plugin-meta">{t("add.pluginRecipes", { recipes: p.recipes.join(", ") })}</p>
      )}
      {p.needs && <p className="rc-add-plugin-meta">{t("add.pluginNeeds", { needs: p.needs })}</p>}
      {p.install && <CopyCommand command={p.install} />}
      <p className="hint">{t("add.pluginRestart")}</p>
    </article>
  );
}
