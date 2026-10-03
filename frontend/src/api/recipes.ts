// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { del, getJSON, needsForce, postJSON, putJSON } from "./client";
import type {
  DeleteWeightsResult,
  Download,
  Policy,
  RecipePlan,
  RecipeWriteResponse,
  RemoveRecipeWeightsResult,
} from "./types";

/* The writes on recipes and weights (ADR-003), free of React so the force
   protocol is testable with a mocked fetch. The Overview's lane actions and
   the Recipes view both use them. */

const path = (name: string) => `/recipes/${encodeURIComponent(name)}`;

export type Forced<T> =
  { ok: true; result: T; forced: boolean } | { ok: false; declined: string };

/* A change that would over-commit a card comes back 409 with "force=true"
   in the detail. The server's detail is the explanation, so it goes to the
   operator verbatim; confirming repeats the call with force. Any other
   failure — including a 409 without that marker, such as two LLMs pinned to
   one card — throws, with no override offered. */
export async function withForce<T>(
  call: (force: boolean) => Promise<T>,
  confirmForce: (detail: string) => Promise<boolean>,
): Promise<Forced<T>> {
  try {
    return { ok: true, result: await call(false), forced: false };
  } catch (err) {
    if (!needsForce(err)) throw err;
    const detail = err.detail as string;
    if (!(await confirmForce(detail))) return { ok: false, declined: detail };
    return { ok: true, result: await call(true), forced: true };
  }
}

/** PUT /recipes/{name}/residency {policy, force}, confirming an over-commit. */
export function setResidency(
  name: string,
  policy: Policy,
  confirmForce: (detail: string) => Promise<boolean>,
): Promise<Forced<RecipeWriteResponse>> {
  return withForce(
    (force) =>
      putJSON<RecipeWriteResponse>(`${path(name)}/residency`, {
        policy,
        force,
      }),
    confirmForce,
  );
}

/** DELETE /recipes/{name}/residency: drop the operator's override, back to the default. */
export function clearResidency(name: string): Promise<RecipeWriteResponse> {
  return del<RecipeWriteResponse>(`${path(name)}/residency`);
}

/** PUT /recipes/{name}/card {device, force}; null device = follow the default card. */
export function setCard(
  name: string,
  device: string | null,
  confirmForce: (detail: string) => Promise<boolean>,
): Promise<Forced<RecipeWriteResponse>> {
  return withForce(
    (force) =>
      putJSON<RecipeWriteResponse>(`${path(name)}/card`, { device, force }),
    confirmForce,
  );
}

/** DELETE /weights/{id}: remove one checkpoint; the recipes that used it stay, uninstalled. */
export function deleteWeights(id: string): Promise<DeleteWeightsResult> {
  return del<DeleteWeightsResult>(`/weights/${encodeURIComponent(id)}`);
}

/** GET /recipes/{name}/plan: what fetching it takes, with the Hub's sizes, access and licence. */
export function planRecipe(name: string): Promise<RecipePlan> {
  return getJSON<RecipePlan>(`${path(name)}/plan`);
}

/** POST /recipes/{name}/fetch: start fetching its missing weights. */
export function fetchRecipe(name: string): Promise<Download> {
  return postJSON<Download>(`${path(name)}/fetch`);
}

/** DELETE /downloads/{id}: stop a fetch; what arrived stays, so the next one resumes. */
export function cancelDownload(id: string): Promise<Download> {
  return del<Download>(`/downloads/${encodeURIComponent(id)}`);
}

/** DELETE /recipes/{name}/weights: its weights, except those another recipe also loads. */
export function removeRecipeWeights(name: string): Promise<RemoveRecipeWeightsResult> {
  return del<RemoveRecipeWeightsResult>(`${path(name)}/weights`);
}
