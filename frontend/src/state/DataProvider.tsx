// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useEffect, type ReactNode } from "react";
import { setModalities } from "../lib/modalities";
import {
  CapabilitiesProvider,
  DownloadsProvider,
  EnginesProvider,
  GpusProvider,
  InstancesProvider,
  PluginsProvider,
  RecipesProvider,
  useCapabilities,
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
                    <PluginsProvider>
                      <CapabilitiesProvider>
                        <ModalitySync />
                        {children}
                      </CapabilitiesProvider>
                    </PluginsProvider>
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

/* Hands the server's modality list (labels, icons, registration order) to
   the module-level registry the icons, labels and series colours read. */
function ModalitySync() {
  const caps = useCapabilities();
  useEffect(() => {
    if (!caps.data) return;
    setModalities(
      Object.entries(caps.data.modalities).map(([name, m]) => ({ name, label: m.label, icon: m.icon })),
    );
  }, [caps.data]);
  return null;
}
