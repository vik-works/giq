// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from "react";
import { getJSON } from "../../api/client";
import type { GpuErasResponse, JobRecord, StatsUsage, UsagePeriod } from "../../api/types";
import { usePathPoll } from "../../lib/usePathPoll";
import { usePoll } from "../../lib/usePoll";
import { SLOW_POLL_MS } from "../../state";

/** How many calls the wall lists. */
export const WALL_LIMIT = 100;

const gpuParam = (gpu: string) => (gpu ? `&gpu=${encodeURIComponent(gpu)}` : "");

/* The usage view's three endpoints. They are the view's own, so they poll
   here rather than in src/state, and only while the view is mounted (the
   router unmounts a view it leaves, which stops the timers): usage figures
   move by the minute, and nobody else reads them. A filter change fetches
   at once (usePathPoll), and `stale` says the figures on screen still
   answer the previous filter. The app filter narrows the usage rollup only:
   /stats/jobs takes no app, so the call wall stays the machine's. */
export function useUsageData(period: UsagePeriod, gpu: string, app = "", enabled = true) {
  const opts = { intervalMs: SLOW_POLL_MS, enabled };
  const eras = usePoll(
    useCallback((signal: AbortSignal) => getJSON<GpuErasResponse>("/stats/gpus/eras", { signal }), []),
    opts,
  );
  const appParam = app ? `&app=${encodeURIComponent(app)}` : "";
  const usagePath = `/stats/usage?period=${period}${gpuParam(gpu)}${appParam}`;
  const jobsPath = `/stats/jobs?limit=${WALL_LIMIT}${gpuParam(gpu)}`;
  const usage = usePathPoll<StatsUsage>(usagePath, opts);
  const jobs = usePathPoll<JobRecord[]>(jobsPath, opts);
  const stale = (!!usage.data && usage.data.path !== usagePath) || (!!jobs.data && jobs.data.path !== jobsPath);
  return {
    eras,
    usage: { ...usage, data: usage.data?.body },
    jobs: { ...jobs, data: jobs.data?.body },
    stale,
  };
}
