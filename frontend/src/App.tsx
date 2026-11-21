// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { lazy, Suspense, type ComponentType, type LazyExoticComponent } from "react";
import { AppShell } from "./components/AppShell";
import { ViewSkeleton } from "./components/ViewSkeleton";
import { loadNamespace, type Namespace } from "./i18n";
import { useHashRoute, type View } from "./lib/useHashRoute";

/* One chunk per view, fetched with its strings the first time the view is
   opened: the control surface should not wait for the sandbox's stream
   parser or the models view's catalog logic. Everything is served by giq
   itself, so a chunk is a local request and the dashboard still works
   offline. */
function view(ns: Namespace, load: () => Promise<{ default: ComponentType }>): LazyExoticComponent<ComponentType> {
  return lazy(async () => {
    const [mod] = await Promise.all([load(), loadNamespace(ns)]);
    return mod;
  });
}

const VIEWS: Record<View, LazyExoticComponent<ComponentType>> = {
  overview: view("overview", () => import("./views/overview")),
  usage: view("usage", () => import("./views/usage")),
  apps: view("apps", () => import("./views/apps")),
  recipes: view("recipes", () => import("./views/recipes")),
  inventory: view("inventory", () => import("./views/inventory")),
  sandbox: view("sandbox", () => import("./views/sandbox")),
};

export default function App() {
  const { view: current } = useHashRoute();
  const Page = VIEWS[current];
  return (
    <AppShell view={current}>
      <Suspense fallback={<ViewSkeleton />}>
        <Page />
      </Suspense>
    </AppShell>
  );
}
