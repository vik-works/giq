// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ApiError, errorText } from "../../api/client";
import { cancelDownload, fetchRecipe, planRecipe } from "../../api/recipes";
import type { Download, RecipeEntry, RecipePlan } from "../../api/types";
import { useNotify } from "../../components/DialogProvider";
import { useDownloads, useRecipes, useStorage, useWeights } from "../../state";

export interface PlanState {
  loading: boolean;
  plan?: RecipePlan;
  error?: string;
  /** The `giq add` command, when the service may not fetch it itself. */
  command?: string;
}

export interface AddActions {
  planOf: (name: string) => PlanState | undefined;
  /** Load the plan: the Hub's sizes, access and licence, and this machine's checks. */
  check: (r: RecipeEntry) => Promise<void>;
  get: (r: RecipeEntry) => Promise<void>;
  cancel: (d: Download) => Promise<void>;
}

/* The Add page's calls. A fetch the service may not run (a read-only model
   store) comes back 409 with the command for the operator, which the row
   then shows instead of a button. When a fetch finishes, the catalog and
   the disk report are fetched again at once, so the recipe moves to "On
   this machine" without waiting for their next minute. */
export function useAddActions(): AddActions {
  const { t } = useTranslation("recipes");
  const notify = useNotify();
  const downloads = useDownloads();
  const recipes = useRecipes();
  const weights = useWeights();
  const storage = useStorage();
  const [plans, setPlans] = useState<ReadonlyMap<string, PlanState>>(new Map());
  const setPlan = useCallback(
    (name: string, state: PlanState) => setPlans((m) => new Map(m).set(name, state)),
    [],
  );

  const finished = useRef<Set<string>>(new Set());
  useEffect(() => {
    const done = (downloads.data?.downloads ?? []).filter(
      (d) => d.state === "done" && !finished.current.has(d.id),
    );
    if (done.length === 0) return;
    for (const d of done) finished.current.add(d.id);
    void Promise.all([recipes.refresh(), weights.refresh(), storage.refresh()]);
  }, [downloads.data, recipes.refresh, weights.refresh, storage.refresh]);

  const check = useCallback(
    async (r: RecipeEntry) => {
      setPlan(r.name, { loading: true });
      try {
        const plan = await planRecipe(r.name);
        setPlan(r.name, {
          loading: false,
          plan,
          command: plan.service_can_fetch ? undefined : plan.command,
        });
      } catch (err) {
        setPlan(r.name, { loading: false, error: errorText(err) });
      }
    },
    [setPlan],
  );

  const get = useCallback(
    async (r: RecipeEntry) => {
      try {
        await fetchRecipe(r.name);
      } catch (err) {
        const detail = err instanceof ApiError ? err.detail : null;
        if (detail && typeof detail === "object" && "command" in detail) {
          setPlans((m) => new Map(m).set(r.name, { ...m.get(r.name), loading: false, command: String(detail.command) }));
        } else {
          await notify({ title: t("add.getFailed", { recipe: r.name }), body: errorText(err) });
        }
      } finally {
        await downloads.refresh();
      }
    },
    [notify, t, downloads.refresh],
  );

  const cancel = useCallback(
    async (d: Download) => {
      try {
        await cancelDownload(d.id);
      } catch (err) {
        await notify({ title: t("add.cancelFailed"), body: errorText(err) });
      } finally {
        await downloads.refresh();
      }
    },
    [notify, t, downloads.refresh],
  );

  return useMemo(
    () => ({ planOf: (name: string) => plans.get(name), check, get, cancel }),
    [plans, check, get, cancel],
  );
}
