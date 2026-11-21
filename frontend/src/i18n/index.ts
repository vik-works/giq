// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import i18n from "i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import { initReactI18next } from "react-i18next";

/* English and German, bundled: giq runs on rigs with no internet, so every
   string ships inside the build rather than loading from a backend.

   Adding a language is one entry here plus a folder of JSON files
   (locales/<code>/<namespace>.json, same keys as en/); the glob below picks
   the files up and the switch in the sidebar lists the entry. */
export const LANGUAGES = [
  { code: "en", label: "EN" },
  { code: "de", label: "DE" },
] as const;

export type LanguageCode = (typeof LANGUAGES)[number]["code"];

export const NAMESPACES = ["common", "overview", "usage", "apps", "recipes", "inventory", "sandbox"] as const;
export type Namespace = (typeof NAMESPACES)[number];

export const LANG_STORAGE_KEY = "giq-lang";

/* The shared namespace ships with the shell; each view's strings travel in
   that view's chunk, loaded next to its code (loadNamespace below), so a
   page pays only for the words it shows. Every language of a namespace
   loads together: switching language never waits on the network. */
const eager = import.meta.glob<Record<string, unknown>>("../locales/*/common.json", {
  eager: true,
  import: "default",
});
const lazy = import.meta.glob<Record<string, unknown>>(["../locales/*/*.json", "!../locales/*/common.json"], {
  import: "default",
});

function parsePath(path: string): [lang: string, ns: string] | null {
  const m = /locales\/([^/]+)\/([^/]+)\.json$/.exec(path);
  return m ? [m[1]!, m[2]!] : null;
}

const resources: Record<string, Record<string, Record<string, unknown>>> = {};
for (const [path, strings] of Object.entries(eager)) {
  const at = parsePath(path);
  if (at) (resources[at[0]] ??= {})[at[1]] = strings;
}

void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    supportedLngs: LANGUAGES.map((l) => l.code),
    // de-AT, de-CH… → de; anything unsupported → en.
    nonExplicitSupportedLngs: true,
    load: "languageOnly",
    fallbackLng: "en",
    ns: ["common"],
    defaultNS: "common",
    interpolation: { escapeValue: false }, // React escapes
    returnNull: false,
    detection: {
      order: ["localStorage", "navigator"],
      lookupLocalStorage: LANG_STORAGE_KEY,
      caches: ["localStorage"],
    },
  });

function syncHtmlLang(lng: string | undefined) {
  if (typeof document !== "undefined" && lng) {
    document.documentElement.lang = lng.split("-")[0] ?? lng;
  }
}
syncHtmlLang(i18n.resolvedLanguage);
i18n.on("languageChanged", () => syncHtmlLang(i18n.resolvedLanguage));

/** Load a view's strings, in every language, before the view renders. */
export async function loadNamespace(ns: Namespace): Promise<void> {
  const files = Object.entries(lazy).filter(([path]) => parsePath(path)?.[1] === ns);
  await Promise.all(
    files.map(async ([path, load]) => {
      const [lang] = parsePath(path)!;
      i18n.addResourceBundle(lang, ns, await load(), true, true);
    }),
  );
}

export default i18n;
