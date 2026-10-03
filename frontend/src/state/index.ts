// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

export { DataProvider } from "./DataProvider";
export {
  LIVE_POLL_MS,
  SLOW_POLL_MS,
  useCapabilities,
  useDownloads,
  useEngines,
  useGpus,
  useInstances,
  usePlugins,
  useRecipes,
  useStatus,
  useStorage,
  useWeights,
} from "./resources";
export type { Resource } from "./createPolledResource";
