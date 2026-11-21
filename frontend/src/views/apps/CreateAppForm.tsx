// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { createApp } from "../../api/apps";
import { errorText } from "../../api/client";
import type { AppCreated } from "../../api/types";
import { Card, Field } from "../../components";
import "./CreateAppForm.css";

/* A name and an optional description. The server owns the naming rule and
   the reserved names, so a refusal is its 400 detail, shown as text under
   the name; the hint only says what a name looks like. */
export function CreateAppForm({ onCreated }: { onCreated: (app: AppCreated) => void }) {
  const { t } = useTranslation("apps");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const created = await createApp(name.trim(), description.trim() || undefined);
      setName("");
      setDescription("");
      onCreated(created);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card kicker={t("create.kicker")}>
      <form className="ap-create" onSubmit={(e) => void submit(e)}>
        <Field label={t("create.name")} hint={t("create.nameHint")} error={error}>
          {(id) => (
            <input
              id={id}
              className="input mono"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoComplete="off"
              spellCheck={false}
              placeholder={t("create.namePlaceholder")}
            />
          )}
        </Field>
        <Field label={t("create.description")}>
          {(id) => (
            <input
              id={id}
              className="input"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder={t("create.descriptionPlaceholder")}
            />
          )}
        </Field>
        <div className="ap-create-actions">
          <button type="submit" className="btn btn-primary" disabled={!name.trim() || busy}>
            {busy ? t("create.creating") : t("create.submit")}
          </button>
        </div>
      </form>
    </Card>
  );
}
