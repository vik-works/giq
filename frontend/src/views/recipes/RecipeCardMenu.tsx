// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { DotsThreeIcon, FlaskIcon, HardDrivesIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import type { RecipeEntry } from "../../api/types";
import { Icon } from "../../components/Icon";
import { Menu } from "../../components/Menu";
import { MenuItem } from "../../components/MenuItem";
import { useModalityLabel } from "../../lib/modalities";
import { sandboxHref, tabForModality } from "../../lib/sandboxLink";
import { usePluginPanels } from "../../plugins/panels";
import { primaryModality } from "../../lib/recipes";
import { sandboxBlock } from "./catalog";

export interface RecipeCardMenuProps {
  r: RecipeEntry;
}

/* The card's ⋯ menu: one entry per thing done *with* the recipe rather than
   a setting of it. Unavailable entries stay listed, disabled, with the reason
   as their description — a missing entry would leave the operator guessing
   whether it exists at all. Its weights are the Inventory's: deleting them is
   an act on files another recipe may share, so it lives there. */
export function RecipeCardMenu({ r }: RecipeCardMenuProps) {
  const { t } = useTranslation("recipes");
  const key = r.name;
  const modalityLabel = useModalityLabel();
  usePluginPanels(); // plugins declare panels; re-render when they arrive
  const served = r.modalities.find((m) => tabForModality(m)) ?? primaryModality(r);
  const tab = tabForModality(served);
  const noTest = sandboxBlock(r);
  return (
    <Menu
      label={<Icon as={DotsThreeIcon} size={16} weight="bold" />}
      ariaLabel={t("menu.label", { key })}
      title={t("common:actions.more")}
      buttonClassName="btn btn-icon btn-ghost model-menu-btn"
    >
      <MenuItem
        icon={<Icon as={FlaskIcon} size={14} />}
        disabled={noTest != null}
        description={noTest ? t(noTest) : t("menu.testHelp", { modality: modalityLabel(served) })}
        // Navigates only; the sandbox preselects the model and waits for Run.
        onSelect={() => {
          if (tab) window.location.hash = sandboxHref(tab, r.name);
        }}
      >
        {t("menu.test")}
      </MenuItem>
      <MenuItem
        icon={<Icon as={HardDrivesIcon} size={14} />}
        disabled={r.weights.length === 0}
        description={r.weights.length ? t("menu.weightsHelp") : t("menu.noWeights")}
        onSelect={() => {
          window.location.hash = `#/inventory?recipe=${encodeURIComponent(r.name)}`;
        }}
      >
        {t("menu.weights")}
      </MenuItem>
    </Menu>
  );
}
