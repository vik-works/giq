// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import * as reactI18next from "react-i18next";
import { del, errorText, getJSON, isAbort, postBlob, postForm, postJSON } from "../api/client";
import { Card } from "../components/Card";
import { EmptyState } from "../components/EmptyState";
import { Field } from "../components/Field";
import { FilePicker } from "../components/FilePicker";
import { Icon } from "../components/Icon";
import { ProgressBar } from "../components/ProgressBar";
import { SegmentedControl } from "../components/SegmentedControl";
import { Tag } from "../components/Tag";
import { WorkerIcon } from "../components/WorkerIcon";
import { useFormat } from "../lib/useFormat";
import { useRunner } from "../lib/useRunner";
import { fileB64 } from "../views/sandbox/shared/fileB64";
import { ModelSelect } from "../views/sandbox/shared/ModelSelect";
import { OutputCard } from "../views/sandbox/shared/OutputCard";
import { RunButton } from "../views/sandbox/shared/RunButton";

/* What a plugin's dashboard module gets from the dashboard (ADR-004 D6).
   A plugin never bundles its own React: two copies break hooks, and a
   second react-i18next would not see the dashboard's strings. Its build
   (the @giq/plugin-ui Vite preset) points react, react/jsx-runtime and
   react-i18next at this object instead, and @giq/plugin-ui's own exports
   come from here too.

   The version is the plugin API's (giq.plugin.API_VERSION): a field that
   changes meaning raises it, and the server leaves out a UI written for
   another. Adding a field does not. */
export const host = {
  version: 1,
  React,
  jsxRuntime,
  reactI18next,
  api: { getJSON, postJSON, postForm, postBlob, del, errorText, isAbort },
  useFormat,
  useRunner,
  components: { Card, EmptyState, Field, FilePicker, Icon, ProgressBar, SegmentedControl, Tag, WorkerIcon },
  /* A sandbox panel's building blocks. Its layout uses the sandbox's
     classes: sbx-panel around it, card elev-sm sbx-form for the form,
     sbx-row / sbx-grow / sbx-narrow inside it, sbx-actions for the buttons. */
  sandbox: { ModelSelect, OutputCard, RunButton, fileB64 },
};

export type GiqHost = typeof host;

declare global {
  // eslint-disable-next-line no-var
  var __GIQ_HOST__: GiqHost | undefined;
}

/** Make the host visible to plugin modules; before the first one is imported. */
export function installHost(): void {
  globalThis.__GIQ_HOST__ = host;
}
