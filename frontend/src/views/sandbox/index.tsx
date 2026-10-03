// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { PageHeader } from "../../components/PageHeader";
import { useHashRoute } from "../../lib/useHashRoute";
import { loadPluginStrings } from "../../plugins/loader";
import { usePluginPanels } from "../../plugins/panels";
import { useRecipes } from "../../state";
import { ChatTab } from "./chat/ChatTab";
import { disabledTabs, sandboxModels, tabOptions } from "./models";
import { parseSandboxHash } from "../../lib/sandboxLink";
import { PluginPanelSlot } from "./PluginPanelSlot";
import { panelId, SandboxTabBar, tabId } from "./SandboxTabBar";
import { DEFAULT_TAB, sandboxTabs, type TabInfo } from "./tabs";
import { ToolsTab } from "./tools/ToolsTab";
import { useModelChoices } from "./useModelChoices";
import { usePreselect } from "./usePreselect";
import { VisionTab } from "./vision/VisionTab";
import "./Sandbox.css";

/* Standard workflows with no knobs beyond the basics, one panel per tab:
   the dashboard's own for LLMs, then the panels plugins bring (ADR-004 D6).
   A panel stays mounted once opened, so switching tabs keeps its prompt,
   its result and a generation still streaming. */
export default function SandboxView() {
  const { t } = useTranslation("sandbox");
  const { navigate } = useHashRoute(); // re-renders on every hash change
  const route = parseSandboxHash(window.location.hash);

  const panels = usePluginPanels();
  const tabs = useMemo(() => sandboxTabs(panels), [panels]);
  // Plugins' tab labels and stylesheets, before their code is needed.
  const [, setStrings] = useState(0);
  useEffect(() => {
    for (const ui of new Set(panels.map((p) => p.ui))) {
      void loadPluginStrings(ui).then(
        () => setStrings((n) => n + 1),
        () => undefined,
      );
    }
  }, [panels]);

  const active = tabs.some((x) => x.id === route.tab) ? route.tab! : DEFAULT_TAB;
  const recipes = useRecipes();
  const models = useMemo(() => sandboxModels(recipes.data), [recipes.data]);
  const disabled = useMemo(() => disabledTabs(models, tabs), [models, tabs]);
  const { value, choose } = useModelChoices(models);
  const missed = usePreselect(route, recipes.data, models, tabs, choose);

  const [visited, setVisited] = useState<Set<string>>(() => new Set([active]));
  if (!visited.has(active)) setVisited(new Set(visited).add(active));

  const panel = (tab: TabInfo) => {
    const props = { options: tabOptions(models, tab), model: value(tab), onModel: (m: string) => choose(tab.id, m) };
    if (tab.plugin) return <PluginPanelSlot panel={tab.plugin} {...props} />;
    switch (tab.id) {
      case "chat":
        return <ChatTab {...props} />;
      case "tools":
        return <ToolsTab available={!disabled.has("tools")} />;
      case "vision":
        return <VisionTab {...props} />;
    }
    return null;
  };

  return (
    <>
      <PageHeader title={t("common:nav.sandbox")} />
      <p className="sbx-lede">{t("lede")}</p>
      <SandboxTabBar tabs={tabs} active={active} disabled={disabled} onSelect={(tab) => navigate("sandbox", tab)} />
      {recipes.error != null && !recipes.data && <p className="warn">{t("catalogFailed")}</p>}
      {missed?.tab === active && <p className="warn">{t("preselectMissed", { model: missed.model })}</p>}
      {tabs
        .filter((tab) => visited.has(tab.id))
        .map((tab) => (
          <div key={tab.id} role="tabpanel" id={panelId(tab.id)} aria-labelledby={tabId(tab.id)} hidden={tab.id !== active}>
            {panel(tab)}
          </div>
        ))}
    </>
  );
}
