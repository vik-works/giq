// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { Highlight, type PrismTheme } from "prism-react-renderer";
import { useTranslation } from "react-i18next";
import { NS } from "./ns";

export interface JsonEditorProps {
  id: string;
  value: string;
  onChange: (v: string) => void;
  rows?: number;
  invalid?: boolean;
}

/* A JSON textarea with line numbers and syntax highlighting via
   prism-react-renderer. The Highlight render is text only (no background,
   no layout of its own): it sits under a transparent textarea in one
   scroll-synced frame, so caret, selection and IME stay native. The theme
   below is literal hex by necessity — prism themes are inline-style
   objects, not CSS — matched once to the Nocturne dark/light ramps; the
   frame, gutter and caret still follow tokens. Invalid JSON gets the
   field-error style on the frame, never a popup. */

const DARK: PrismTheme = {
  plain: { color: "#e8e6e3" },
  styles: [
    { types: ["property", "attr-name"], style: { color: "#8ecae6" } },
    { types: ["string", "char"], style: { color: "#75ce8a" } },
    { types: ["number", "boolean"], style: { color: "#e2ad4a" } },
    { types: ["null", "nil"], style: { color: "#fc978f" } },
    { types: ["punctuation", "operator"], style: { color: "#9a958e" } },
  ],
};

const LIGHT: PrismTheme = {
  plain: { color: "#1f2937" },
  styles: [
    { types: ["property", "attr-name"], style: { color: "#1d4ed8" } },
    { types: ["string", "char"], style: { color: "#1b9246" } },
    { types: ["number", "boolean"], style: { color: "#9e710e" } },
    { types: ["null", "nil"], style: { color: "#c34f4b" } },
    { types: ["punctuation", "operator"], style: { color: "#6b7280" } },
  ],
};

const themeFor = (dark: boolean): PrismTheme => (dark ? DARK : LIGHT);

export function JsonEditor({ id, value, onChange, rows = 8, invalid }: JsonEditorProps) {
  const { t } = useTranslation(NS);
  const [error, setError] = useState<string | null>(null);
  const [dark, setDark] = useState(
    () =>
      typeof window !== "undefined" &&
      (document.documentElement.dataset.theme === "dark" ||
        (document.documentElement.dataset.theme !== "light" &&
          window.matchMedia("(prefers-color-scheme: dark)").matches)),
  );
  const check = (v: string) => {
    onChange(v);
    if (v.trim() === "") return setError(null);
    try {
      JSON.parse(v);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <div className={`pl-giq-imajev-json${invalid || error ? " pl-giq-imajev-json-invalid" : ""}`}>
      <div className="pl-giq-imajev-json-edit">
        <Highlight theme={themeFor(dark)} language="json" code={value}>
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre className="pl-giq-imajev-json-hl" aria-hidden="true">
              {tokens.map((line, i) => {
                const { key: _lk, ...lineProps } = getLineProps({ line }) as {
                  key?: React.Key;
                  [k: string]: unknown;
                };
                void _lk;
                const empty = line.length === 1 && line[0]?.content === "";
                return (
                  <div key={i} {...lineProps}>
                    {empty ? (
                      <span>{"\u200b"}</span>
                    ) : (
                      line.map((token, k) => {
                        const { key: _tk, ...tokenProps } = getTokenProps({ token }) as {
                          key?: React.Key;
                          [k: string]: unknown;
                        };
                        void _tk;
                        return <span key={k} {...tokenProps} />;
                      })
                    )}
                  </div>
                );
              })}
            </pre>
          )}
        </Highlight>
        <textarea
          id={id}
          className="pl-giq-imajev-json-input"
          rows={rows}
          wrap="off"
          value={value}
          spellCheck={false}
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
          aria-invalid={invalid || !!error || undefined}
          aria-describedby={error ? `${id}-err` : undefined}
          onChange={(e) => check(e.target.value)}
          onFocus={() =>
            setDark(
              document.documentElement.dataset.theme === "dark" ||
                (document.documentElement.dataset.theme !== "light" &&
                  window.matchMedia("(prefers-color-scheme: dark)").matches),
            )
          }
        />
      </div>
      {error && (
        <p className="field-error" role="alert" id={`${id}-err`}>
          {t("decide.jsonError", { error })}
        </p>
      )}
    </div>
  );
}
