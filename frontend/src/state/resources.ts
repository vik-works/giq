// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type {
  DownloadsResponse,
  EnginesResponse,
  GpusResponse,
  InstancesResponse,
  PluginsResponse,
  RecipesResponse,
  Status,
  StorageResponse,
  WeightsResponse,
} from "../api/types";
import { createPolledResource } from "./createPolledResource";

/** Poll cadence of /status and /gpus; the page header quotes it. */
export const LIVE_POLL_MS = 3000;
/** The catalog and disk report change on operator action, not by the second. */
export const SLOW_POLL_MS = 60_000;

/** GET /status every 3 s: service state, queue, pause, access posture, version. */
export const [StatusProvider, useStatus] = createPolledResource<Status>("Status", "/status", LIVE_POLL_MS);

/** GET /gpus every 3 s: per-card telemetry. */
export const [GpusProvider, useGpus] = createPolledResource<GpusResponse>("Gpus", "/gpus", LIVE_POLL_MS);

/** GET /recipes every 60 s: every recipe with its fit, residency and card. Call refresh() after changing one. */
export const [RecipesProvider, useRecipes] = createPolledResource<RecipesResponse>(
  "Recipes",
  "/recipes",
  SLOW_POLL_MS,
);

/** GET /instances every 3 s: the recipes running on each card. */
export const [InstancesProvider, useInstances] = createPolledResource<InstancesResponse>(
  "Instances",
  "/instances",
  LIVE_POLL_MS,
);

/** GET /weights every 60 s: every checkpoint, once each. Call refresh() after deleting one. */
export const [WeightsProvider, useWeights] = createPolledResource<WeightsResponse>(
  "Weights",
  "/weights",
  SLOW_POLL_MS,
);

/** GET /storage every 60 s: weights on disk and per-mount usage. Call refresh() after deleting weights. */
export const [StorageProvider, useStorage] = createPolledResource<StorageResponse>(
  "Storage",
  "/storage",
  SLOW_POLL_MS,
);

/** GET /engines once: engine binaries and their build strings (they change only on redeploy). */
export const [EnginesProvider, useEngines] = createPolledResource<EnginesResponse>(
  "Engines",
  "/engines",
  0,
);

/** GET /downloads every 3 s: the fetches since the service started, with their progress. */
export const [DownloadsProvider, useDownloads] = createPolledResource<DownloadsResponse>(
  "Downloads",
  "/downloads",
  LIVE_POLL_MS,
);

/** GET /plugins once: installed plugins and the curated ones that are not (a restart changes it). */
export const [PluginsProvider, usePlugins] = createPolledResource<PluginsResponse>(
  "Plugins",
  "/plugins",
  0,
);
