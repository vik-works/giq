// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { sfxTask } from "./sfxTask";

const form = { prompt: "  rain on a tin roof ", duration: "", steps: "", guidance: "", seed: "" };

describe("sfxTask", () => {
  it("leaves what the form leaves empty to the recipe", () => {
    expect(sfxTask(form)).toEqual({ id: "sbx-sfx", prompt: "rain on a tin roof" });
  });

  it("sends the numbers the form names", () => {
    const task = sfxTask({ ...form, duration: "4.5", steps: "50", guidance: "3", seed: "0" });
    expect(task).toMatchObject({ duration: 4.5, steps: 50, guidance: 3, seed: 0 });
  });
});
