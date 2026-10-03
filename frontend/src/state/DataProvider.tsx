// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";
import {
  DownloadsProvider,
  EnginesProvider,
  GpusProvider,
  InstancesProvider,
  PluginsProvider,
  RecipesProvider,
  StatusProvider,
  StorageProvider,
  WeightsProvider,
} from "./resources";

/** Mounts every shared poller once, around the whole app. */
export function DataProvider({ children }: { children: ReactNode }) {
  return (
    <StatusProvider>
      <GpusProvider>
        <InstancesProvider>
          <RecipesProvider>
            <WeightsProvider>
              <StorageProvider>
                <EnginesProvider>
                  <DownloadsProvider>
                    <PluginsProvider>{children}</PluginsProvider>
                  </DownloadsProvider>
                </EnginesProvider>
              </StorageProvider>
            </WeightsProvider>
          </RecipesProvider>
        </InstancesProvider>
      </GpusProvider>
    </StatusProvider>
  );
}
