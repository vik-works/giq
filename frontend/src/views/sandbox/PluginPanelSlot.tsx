// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { Component, lazy, Suspense, useMemo, type ComponentType, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { PanelProps } from "../../../plugin-ui/types";
import { loadPluginModule } from "../../plugins/loader";
import type { PluginPanel } from "../../plugins/panels";

export interface PluginPanelSlotProps extends PanelProps {
  panel: PluginPanel;
}

/* A plugin's panel: its module is imported the first time the tab opens,
   and a panel that does not load, or throws while it renders, is replaced by
   a note saying so — one plugin's bug does not take the sandbox down. */
export function PluginPanelSlot({ panel, ...props }: PluginPanelSlotProps) {
  const Panel = useMemo(
    () =>
      lazy(async (): Promise<{ default: ComponentType<PanelProps> }> => {
        const mod = await loadPluginModule(panel.ui);
        const found = mod.panels?.[panel.id];
        if (!found) throw new Error(`${panel.plugin} declares panel ${panel.id} but does not export it`);
        return { default: found };
      }),
    [panel],
  );
  return (
    <PanelBoundary plugin={panel.plugin}>
      <Suspense fallback={<Loading />}>
        <Panel {...props} />
      </Suspense>
    </PanelBoundary>
  );
}

function Loading() {
  const { t } = useTranslation("sandbox");
  return <p className="hint">{t("plugin.loading")}</p>;
}

function Failed({ plugin, error }: { plugin: string; error: string }) {
  const { t } = useTranslation("sandbox");
  return (
    <p className="error-box" role="alert">
      {t("plugin.failed", { plugin, error })}
    </p>
  );
}

class PanelBoundary extends Component<{ plugin: string; children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };

  static getDerivedStateFromError(err: unknown) {
    return { error: err instanceof Error ? err.message : String(err) };
  }

  render() {
    if (this.state.error != null) return <Failed plugin={this.props.plugin} error={this.state.error} />;
    return this.props.children;
  }
}
