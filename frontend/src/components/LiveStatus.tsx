// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from "react-i18next";
import { stateMessage } from "../lib/stateMessage";
import { useFormat } from "../lib/useFormat";
import { LIVE_POLL_MS, useStatus } from "../state";
import { LiveDot } from "./LiveDot";
import { useModalityLabel } from "../lib/modalities";

/* The page header's pulse: live while /status answers, neutral when
   serving is paused, red when giq stops answering. What giq is doing rides
   alongside, said in the page's language from /status's facts; the
   server's own English sentence is the tooltip. */
export function LiveStatus() {
  const { t } = useTranslation();
  const modalityLabel = useModalityLabel();
  const f = useFormat();
  const { data, error, loading } = useStatus();
  if (error) return <LiveDot state="down" label={t("live.unreachable")} />;
  if (loading || !data) return <LiveDot state="connecting" label={t("live.connecting")} />;
  if (data.paused) return <LiveDot state="paused" label={t("live.paused")} />;
  const msg = stateMessage(data);
  const p = msg.params;
  const text = t(`stateMessage.${msg.key}`, {
    ...p,
    modality: typeof p.modality === "string" ? modalityLabel(p.modality) : undefined,
    free: typeof p.free === "number" ? f.gb(p.free) : undefined,
  });
  return (
    <span className="live-status">
      <LiveDot state="live" label={t("live.polling", { seconds: LIVE_POLL_MS / 1000 })} />
      <span className="live-message" title={data.state_message ?? undefined}>
        {text}
      </span>
    </span>
  );
}
