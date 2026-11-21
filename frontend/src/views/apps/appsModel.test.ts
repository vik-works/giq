// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import type { AppEntry } from "../../api/types";
import { appStatus, isoSeconds, keySnippets, sortApps, usageHref } from "./appsModel";

const app = (name: string, revoked: string | null = null): AppEntry => ({
  name,
  description: null,
  created: "2026-11-20T10:12:00Z",
  revoked,
});

describe("apps", () => {
  it("lists active apps first, each group by name", () => {
    const sorted = sortApps([app("zeta"), app("beta", "2026-11-21T08:00:00Z"), app("alpha")]);
    expect(sorted.map((a) => a.name)).toEqual(["alpha", "zeta", "beta"]);
    expect(sorted.map(appStatus)).toEqual(["active", "active", "revoked"]);
  });
  it("reads the server's ISO dates as epoch seconds", () => {
    expect(isoSeconds("2026-11-20T10:12:00Z")).toBe(Date.UTC(2026, 10, 20, 10, 12) / 1000);
    expect(isoSeconds(null)).toBeNull();
    expect(isoSeconds("not a date")).toBeNull();
  });
  it("links to the usage view filtered to the app", () => {
    expect(usageHref("photo-sorter")).toBe("#/usage?app=photo-sorter");
  });
  it("puts the key where the OpenAI SDKs and curl send it", () => {
    const s = keySnippets("http://rig:8084/", "giq-abc");
    expect(s.python).toContain('base_url="http://rig:8084/v1"');
    expect(s.python).toContain('api_key="giq-abc"');
    expect(s.curl).toBe('curl http://rig:8084/v1/models -H "Authorization: Bearer giq-abc"');
  });
});
