// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import i18n from "../i18n";
import type { PluginModule, ServedUi } from "../../plugin-ui/types";
import { installHost } from "./host";

/* Loading a plugin's UI (ADR-004 D6). Its strings and stylesheet load when
   the sandbox opens, so the tabs carry their labels; its code when one of
   its panels is first shown. Each loads once per page. Strings go into the
   namespace plugin-<name>; a language the plugin does not ship falls back
   to its English. */

const strings = new Map<string, Promise<void>>();
const modules = new Map<string, Promise<PluginModule>>();

export const namespaceOf = (plugin: string) => `plugin-${plugin}`;

async function fetchJSON(url: string): Promise<Record<string, unknown>> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return (await res.json()) as Record<string, unknown>;
}

/** The plugin's strings (English, and the page's language) and stylesheets. */
export function loadPluginStrings(ui: ServedUi): Promise<void> {
  let pending = strings.get(ui.plugin);
  if (!pending) {
    pending = (async () => {
      for (const href of ui.styles ?? []) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = ui.base + href;
        link.dataset.giqPlugin = ui.plugin;
        document.head.appendChild(link);
      }
      const load = async (lang: string) => {
        const file = ui.locales?.[lang];
        if (!file || i18n.hasResourceBundle(lang, namespaceOf(ui.plugin))) return;
        i18n.addResourceBundle(lang, namespaceOf(ui.plugin), await fetchJSON(ui.base + file), true, true);
      };
      // A language switched to later loads then; until it arrives, English shows.
      i18n.on("languageChanged", (lang) => void load(lang).catch(() => undefined));
      await Promise.all(["en", i18n.resolvedLanguage ?? i18n.language].map(load));
    })();
    strings.set(ui.plugin, pending);
  }
  return pending;
}

/** The plugin's module, imported from where the server serves it. */
export function loadPluginModule(ui: ServedUi): Promise<PluginModule> {
  let pending = modules.get(ui.plugin);
  if (!pending) {
    installHost();
    pending = loadPluginStrings(ui).then(
      () => import(/* @vite-ignore */ ui.base + ui.module) as Promise<PluginModule>,
    );
    modules.set(ui.plugin, pending);
    // A failed import is not cached: the next visit tries again.
    pending.catch(() => modules.delete(ui.plugin));
  }
  return pending;
}
