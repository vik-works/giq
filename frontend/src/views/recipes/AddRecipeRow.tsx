// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { DownloadSimpleIcon, MagnifyingGlassIcon } from "@phosphor-icons/react";
import { useTranslation } from "react-i18next";
import type { Download, RecipeEntry } from "../../api/types";
import { Icon } from "../../components/Icon";
import { Tag, type TagTone } from "../../components/Tag";
import { WorkerIcon } from "../../components/WorkerIcon";
import { primaryModality } from "../../lib/recipes";
import { useFormat } from "../../lib/useFormat";
import { caveat, isActive } from "./adding";
import { CopyCommand } from "./CopyCommand";
import { DownloadProgress } from "./DownloadProgress";
import { PlanChecks } from "./PlanChecks";
import type { AddActions } from "./useAddActions";
import "./AddRecipeRow.css";

const TONE: Record<string, TagTone> = { fetchable: "accent", manual: "neutral", unfit: "critical" };

export interface AddRecipeRowProps {
  r: RecipeEntry;
  download: Download | undefined;
  actions: AddActions;
}

/* One recipe that is not on this machine. Nothing is fetched on sight: the
   first step is the plan (size, access, licence, disk), and only a plan that
   finds nothing in the way offers the button that fetches. A manual recipe
   says where its files go; a fetch the service may not run hands over the
   command instead. */
export function AddRecipeRow({ r, download, actions }: AddRecipeRowProps) {
  const { t } = useTranslation("recipes");
  const fmt = useFormat();
  const state = actions.planOf(r.name);
  const plan = state?.plan;
  const running = isActive(download);
  const failed = download?.state === "failed" ? download : undefined;
  const why = caveat(r);
  return (
    <article className="card elev-sm rc-add-row" aria-busy={running || state?.loading}>
      <header className="rc-add-row-head">
        <WorkerIcon worker={primaryModality(r)} size={16} labelled />
        <h3 className="rc-add-row-name">{r.name}</h3>
        <Tag tone={TONE[r.availability] ?? "neutral"}>{t(`add.availability.${r.availability}`)}</Tag>
      </header>
      <p className="rc-add-row-meta">
        {r.measured ? t("card.vram", { vram: fmt.gb(r.vram_gb) }) : t("card.vramEstimated", { vram: fmt.gb(r.vram_gb) })}
        {" · "}
        <span className="mono">{r.engine}</span>
        {r.detail && <> · {r.detail}</>}
      </p>
      {plan ? <PlanChecks checks={plan.checks} /> : why && <p className="rc-add-row-why">{why}</p>}
      {state?.error && <p className="field-error">{state.error}</p>}
      {failed && <p className="field-error">{t("add.failed", { error: failed.error ?? "" })}</p>}
      {running && download ? (
        <DownloadProgress d={download} onCancel={() => void actions.cancel(download)} />
      ) : state?.command ? (
        <div className="rc-add-row-operator">
          <p className="hint">{t("add.operator")}</p>
          <CopyCommand command={state.command} />
        </div>
      ) : (
        r.availability === "fetchable" && (
          <div className="rc-add-row-actions">
            {plan?.can_fetch ? (
              <button type="button" className="btn btn-primary btn-sm" onClick={() => void actions.get(r)}>
                <Icon as={DownloadSimpleIcon} size={14} />
                {plan.download_bytes
                  ? t("add.get", { size: fmt.bytes(plan.download_bytes) })
                  : t("add.getUnknown")}
              </button>
            ) : (
              !plan && (
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={state?.loading}
                  onClick={() => void actions.check(r)}
                >
                  <Icon as={MagnifyingGlassIcon} size={14} />
                  {state?.loading ? t("add.checking") : t("add.checkPlan")}
                </button>
              )
            )}
          </div>
        )
      )}
    </article>
  );
}
