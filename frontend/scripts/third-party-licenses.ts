// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

// Emits THIRD_PARTY_LICENSES.txt next to the built dashboard: every npm
// package whose code or assets end up in the production bundle, with its
// version, SPDX licence and full licence text. The dashboard ships inside the
// Python wheel, and MIT and OFL-1.1 both require their text to travel with
// every copy. The build fails on a package with no licence, no licence text,
// or a licence outside ALLOWED — a new dependency is a licensing decision,
// not something to find out after a release.

import fs from "node:fs";
import path from "node:path";
import license from "rollup-plugin-license";
import type { Plugin } from "vite";

export const ALLOWED = new Set([
  "MIT",
  "ISC",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "Apache-2.0",
  "OFL-1.1",
  "0BSD",
]);

const FILE_NAME = "THIRD_PARTY_LICENSES.txt";
const LICENSE_FILE = /^(licen[cs]e|copying|ofl)(\.(md|txt|markdown))?$/i;

interface Entry {
  name: string;
  version: string;
  license: string;
  text: string;
}

// An SPDX expression passes when every identifier in it is allowed; "OR"
// alternatives would be enough on their own, but every package in the bundle
// today names a single licence, so the strict reading costs nothing.
function allowed(spdx: string | null): boolean {
  if (!spdx) return false;
  const ids = spdx.replace(/[()]/g, " ").split(/\s+(?:OR|AND)\s+|\s+/).filter(Boolean);
  return ids.length > 0 && ids.every((id) => ALLOWED.has(id));
}

function readLicenseText(dir: string): string | null {
  const file = fs.readdirSync(dir).find((f) => LICENSE_FILE.test(f));
  return file ? fs.readFileSync(path.join(dir, file), "utf-8").trim() : null;
}

// The package.json that names the package a node_modules file belongs to
// (skipping the nameless {"type": "module"} stubs some packages nest in dist/).
function owningPackage(file: string): { dir: string; pkg: Record<string, unknown> } | null {
  let dir = path.dirname(file);
  while (dir.includes("node_modules")) {
    const manifest = path.join(dir, "package.json");
    if (fs.existsSync(manifest)) {
      const pkg = JSON.parse(fs.readFileSync(manifest, "utf-8"));
      if (pkg.name && pkg.version) return { dir, pkg };
    }
    dir = path.dirname(dir);
  }
  return null;
}

function check(entry: { name: string; license: string | null; text: string | null }): void {
  if (!entry.license) throw new Error(`${entry.name}: bundled, but declares no licence`);
  if (!allowed(entry.license)) {
    throw new Error(
      `${entry.name}: licence ${entry.license} is not in the allowlist ` +
        `(${[...ALLOWED].join(", ")}) — see "Third-party licences" in docs/development.md`,
    );
  }
  if (!entry.text) throw new Error(`${entry.name}: bundled, but ships no LICENSE/COPYING/OFL file`);
}

function render(entries: Entry[], bundle: string): string {
  const sorted = [...entries].sort((a, b) => a.name.localeCompare(b.name));
  const header = [
    `Third-party software bundled in ${bundle}.`,
    "giq itself is licensed under Apache-2.0; see LICENSE and NOTICE.",
    "",
    ...sorted.map((e) => `  ${e.name}@${e.version} (${e.license})`),
  ].join("\n");
  const bodies = sorted.map(
    (e) => `${"=".repeat(78)}\n${e.name}@${e.version}\nLicense: ${e.license}\n${"-".repeat(78)}\n\n${e.text}\n`,
  );
  return `${header}\n\n${bodies.join("\n")}`;
}

/**
 * rollup-plugin-license finds the packages behind the rendered JavaScript.
 * It skips modules that render to nothing, which is what a CSS import becomes
 * (@fontsource's @font-face sheet and the woff2 files it pulls in), so the
 * companion plugin collects the packages of every CSS module in the graph and
 * both sets go into one file. ``bundle`` names what the file is about: the
 * dashboard by default, or a plugin panel that bundles packages of its own
 * (a panel is a separate build, so the dashboard's file does not cover it).
 */
export function thirdPartyLicenses(bundle = "the giq dashboard (giq/static/ui/)"): Plugin[] {
  const cssPackages = new Map<string, Entry>();
  let report: string | null = null;

  const css: Plugin = {
    name: "giq:third-party-css",
    apply: "build",
    buildEnd() {
      for (const id of this.getModuleIds()) {
        const file = id.split("?")[0] ?? id;
        if (!file.includes("/node_modules/") || !file.endsWith(".css")) continue;
        const owner = owningPackage(file);
        if (!owner) throw new Error(`${file}: bundled CSS outside any package`);
        const name = String(owner.pkg.name);
        if (cssPackages.has(name)) continue;
        const entry = {
          name,
          version: String(owner.pkg.version),
          license: typeof owner.pkg.license === "string" ? owner.pkg.license : null,
          text: readLicenseText(owner.dir),
        };
        check(entry);
        cssPackages.set(name, entry as Entry);
      }
    },
  };

  const js = license({
    thirdParty: {
      includePrivate: true,
      multipleVersions: true,
      allow: {
        test: (dep) => allowed(dep.license),
        failOnUnlicensed: true,
        failOnViolation: true,
      },
      output: (deps) => {
        const entries = new Map<string, Entry>();
        for (const dep of deps) {
          const entry = {
            name: dep.name ?? "(unnamed)",
            license: dep.license,
            text: dep.licenseText?.trim() ?? null,
          };
          check(entry);
          entries.set(`${entry.name}@${dep.version}`, { ...entry, version: dep.version ?? "?" } as Entry);
        }
        for (const e of cssPackages.values()) entries.set(`${e.name}@${e.version}`, e);
        report = render([...entries.values()], bundle);
      },
    },
  }) as Plugin;

  // Emitted as an asset rather than written to disk, so it lands wherever
  // Vite writes the bundle and survives emptyOutDir.
  const emit: Plugin = {
    name: "giq:third-party-emit",
    apply: "build",
    generateBundle: {
      order: "post",
      handler() {
        if (report === null) throw new Error("rollup-plugin-license produced no third-party report");
        this.emitFile({ type: "asset", fileName: FILE_NAME, source: report });
      },
    },
  };

  return [css, { ...js, apply: "build" }, emit];
}
