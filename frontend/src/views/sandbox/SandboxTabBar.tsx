// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { WorkerIcon } from "../../components/WorkerIcon";
import { namespaceOf } from "../../plugins/loader";
import type { TabInfo } from "./tabs";
import "./SandboxTabBar.css";

export interface SandboxTabBarProps {
  tabs: TabInfo[];
  active: string;
  disabled: Set<string>;
  onSelect: (tab: string) => void;
}

export const tabId = (tab: string) => `sbx-tab-${tab}`;
export const panelId = (tab: string) => `sbx-panel-${tab}`;

/* ARIA tabs: one tab stop, arrow keys move between the enabled tabs (and
   select them, since each is a cheap view switch), Home/End jump. A tab
   with nothing to run is disabled with a title saying why. */
export function SandboxTabBar({ tabs, active, disabled, onSelect }: SandboxTabBarProps) {
  const { t } = useTranslation("sandbox");
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const enabled = tabs.map((x) => x.id).filter((x) => !disabled.has(x));
  /* A plugin's tab is labelled in its own strings (plugin-<name>), loaded
     when the sandbox opens; until they arrive, the id stands in. */
  const label = (tab: TabInfo) =>
    tab.plugin ? t(tab.plugin.label, { ns: namespaceOf(tab.plugin.plugin), defaultValue: tab.id }) : t(`tab.${tab.id}`);

  // On a phone the strip scrolls sideways; keep the open tab in view (deep links land off-screen otherwise).
  useEffect(() => {
    refs.current[active]?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [active]);

  const onKey = (e: KeyboardEvent) => {
    const i = enabled.indexOf(active);
    const next =
      e.key === "ArrowRight" ? enabled[(i + 1) % enabled.length]
      : e.key === "ArrowLeft" ? enabled[(i - 1 + enabled.length) % enabled.length]
      : e.key === "Home" ? enabled[0]
      : e.key === "End" ? enabled[enabled.length - 1]
      : undefined;
    if (!next) return;
    e.preventDefault();
    onSelect(next);
    refs.current[next]?.focus();
  };

  return (
    <div className="sbx-tabs" role="tablist" aria-label={t("tabsLabel")} onKeyDown={onKey}>
      {tabs.map((tab) => {
        const off = disabled.has(tab.id);
        const on = tab.id === active;
        return (
          <button
            key={tab.id}
            ref={(el) => {
              refs.current[tab.id] = el;
            }}
            type="button"
            role="tab"
            id={tabId(tab.id)}
            aria-selected={on}
            aria-controls={panelId(tab.id)}
            tabIndex={on ? 0 : -1}
            disabled={off}
            title={off ? t("tabDisabled") : undefined}
            className={`sbx-tab${on ? " sbx-tab-active" : ""}`}
            onClick={() => onSelect(tab.id)}
          >
            <WorkerIcon worker={tab.icon} size={14} colored={on} />
            {label(tab)}
          </button>
        );
      })}
    </div>
  );
}
