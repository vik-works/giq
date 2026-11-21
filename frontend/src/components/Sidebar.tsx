// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import {
  ChartBarIcon,
  FlaskIcon,
  HardDrivesIcon,
  KeyIcon,
  PulseIcon,
  StackIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import { hrefFor, type View } from "../lib/useHashRoute";
import { useFormat } from "../lib/useFormat";
import { useStatus } from "../state";
import { AccessChip } from "./AccessChip";
import { Icon } from "./Icon";
import { LanguageSwitch } from "./LanguageSwitch";
import { ThemeSwitch } from "./ThemeSwitch";
import "./Sidebar.css";

const NAV: { view: View; icon: PhosphorIcon }[] = [
  { view: "overview", icon: PulseIcon },
  { view: "recipes", icon: StackIcon },
  { view: "inventory", icon: HardDrivesIcon },
  { view: "usage", icon: ChartBarIcon },
  { view: "apps", icon: KeyIcon },
  { view: "sandbox", icon: FlaskIcon },
];

export interface SidebarProps {
  view: View;
  /** Narrow screens: whether the drawer is shown. */
  open: boolean;
  /** Called after a nav link is followed (closes the drawer). */
  onNavigate: () => void;
  id?: string;
}

export function Sidebar({ view, open, onNavigate, id }: SidebarProps) {
  const { t } = useTranslation();
  const f = useFormat();
  const status = useStatus().data;
  return (
    <aside className={`sidebar${open ? " sidebar-open" : ""}`} id={id}>
      <a className="sidebar-brand" href={hrefFor("overview")} onClick={onNavigate}>
        <span className="brand-mark">
          giq<span className="brand-cursor">_</span>
        </span>
        <span className="brand-sub">{t("app.subtitle")}</span>
      </a>
      <nav className="sidebar-nav" aria-label={t("nav.label")}>
        {NAV.map(({ view: v, icon }) => (
          <a
            key={v}
            href={hrefFor(v)}
            className="sidebar-link"
            aria-current={v === view ? "page" : undefined}
            onClick={onNavigate}
          >
            <Icon as={icon} size={16} />
            {t(`nav.${v}`)}
          </a>
        ))}
      </nav>
      <div className="sidebar-footer">
        <AccessChip />
        <div className="sidebar-switches">
          <LanguageSwitch />
          <ThemeSwitch />
        </div>
        {status && (
          <div className="sidebar-meta">
            {t("footer.version", { version: status.version })}
            {" · "}
            {t("footer.uptime", { uptime: f.uptime(status.uptime_s) })}
          </div>
        )}
      </div>
    </aside>
  );
}
