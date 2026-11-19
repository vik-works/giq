// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* Pure: the sandbox form as a text2sfx task. */

/** The form's values; an empty string leaves the field to the recipe or the model. */
export interface SfxForm {
  prompt: string;
  duration: string;
  steps: string;
  guidance: string;
  seed: string;
}

/** The form as a text2sfx task: only the fields the user filled in. */
export function sfxTask(form: SfxForm): Record<string, unknown> {
  const task: Record<string, unknown> = { id: "sbx-sfx", prompt: form.prompt.trim() };
  for (const key of ["duration", "steps", "guidance", "seed"] as const) {
    if (form[key].trim() !== "") task[key] = Number(form[key]);
  }
  return task;
}
