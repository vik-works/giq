// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it } from "vitest";
import { modalities, modalityMeta, registrationIndex, setModalities } from "./modalities";
import { workerColor } from "./series";

describe("the modality registry", () => {
  afterEach(() => setModalities([]));

  it("keeps the server's order and metadata", () => {
    setModalities([
      { name: "llm", label: "LLM", icon: "chat" },
      { name: "multiview", label: "Multiview", icon: "cube" },
    ]);
    expect(registrationIndex("multiview")).toBe(1);
    expect(modalityMeta("multiview")?.label).toBe("Multiview");
    expect(registrationIndex("nope")).toBe(-1);
  });

  it("does not replace an equal list, so readers do not re-render", () => {
    const list = [{ name: "llm", label: "LLM", icon: "chat" }];
    setModalities(list);
    const before = modalities();
    setModalities([{ ...list[0]! }]);
    expect(modalities()).toBe(before);
  });

  it("colours a plugin's modality by its place, a known one by its own slot", () => {
    setModalities([
      { name: "llm", label: "LLM", icon: "chat" },
      { name: "multiview", label: "Multiview", icon: "cube" },
    ]);
    expect(workerColor("llm")).toBe("var(--s-llm)");
    expect(workerColor("multiview")).toBe("var(--series-2)");
    expect(workerColor("unregistered")).toBe("var(--series-other)");
  });
});
