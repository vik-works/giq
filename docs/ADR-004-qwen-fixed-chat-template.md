<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# ADR-004: Fixed Qwen chat templates via `params.chat_template_file`

**Status:** Accepted — implemented as opt-in (`support-chat-templates`)
**Date:** 2026-10-03
**Authors:** giq maintainers
**Amends:** nothing; uses the `params` mechanism of [ADR-002](ADR-002-model-instances.md) D2 and the terms of [ADR-003](ADR-003-domain.md)

## Context

giq serves Qwen 3.5/3.6/3.8 recipes on both LLM engines and uses each
checkpoint's own chat template: llama-server runs `--jinja` over the GGUF's
embedded template, vLLM over the checkpoint's `chat_template.jinja`. No
recipe overrode it — there was no knob.

The stock Qwen 3.8 template (as shipped in `nvidia-Qwen3.8-27B-NVFP4`,
md5 `519239a4…`) has three behaviours that hurt giq's agentic callers:

1. **Reasoning effort defaults to `xhigh`.** The template injects the deep
   reasoning instruction unless the caller passes `reasoning_effort`. Every
   plain turn pays ~42 prompt tokens for instruction it did not ask for, and
   long thoughts burn the generation budget before the answer starts — the
   empty-answer `finish_reason: length` failure the `reasoning_budget_tokens`
   deadline and the loop guard already fight at other layers.
2. **History with JSON-string tool arguments crashes rendering.** Assistant
   turns whose `tool_calls[].function.arguments` arrive as a serialized JSON
   string (the standard OpenAI proxy wire shape) hit `tool_call.arguments |
   items` on a string: `TypeError: Can only get item pairs from a mapping`.
   Offline this raises; through vLLM the arguments are coerced to an empty
   object (`Tool call 'get_weather' has arguments that are not valid JSON …
   coercing to an empty object`), so the model re-emits the same tool call
   instead of using the tool result — redundant calls, longer histories,
   more looping surface.
3. **No per-request override path through giq.** vLLM 0.30.0 accepts a
   per-request `chat_template` string, but only with
   `--trust-request-chat-template` (default off; without it a 400), and
   giq's `ChatCompletionRequest` is `extra="ignore"`, so a caller-supplied
   template is silently dropped. A fleet template is a server decision, not
   a client request field.

[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
(v22.5, Apache-2.0, same licence as Qwen) is a drop-in Jinja file covering
Qwen 3.5/3.6/3.8 that addresses exactly these: default effort `medium`
(zero injected tokens), universal mapping/JSON-string argument handling,
`enable_thinking=false` fast mode, strict tool-call directives, minijinja
safe filters. Its author suite (105 checks incl. a property fuzzer) passes
in giq's own venv.

## Decision

- **One opt-in recipe parameter, both engines:** `params.chat_template_file`
  — a Jinja file resolved like `weights.path` (relative to `GIQ_MODELS_DIR`,
  `~`/absolute as written). llama.cpp spells it `--chat-template-file`
  (alongside the existing `--jinja`); vLLM spells it `--chat-template`.
  Unset (every built-in) keeps the checkpoint's own template.
- **No per-request template field.** The API layer stays as it is; a
  template is deployment configuration, versioned with the recipe.
- **No fleet default yet.** The fixed prompt changes system tokens, so
  VRAM/DRY/budget figures measured against the native template need
  re-measuring per recipe. Operators opt in per recipe; a default flip is a
  later decision with its own numbers.

## Verification (measured, this machine)

Isolated `GIQ_HOME`, IQ2_XS Qwen3.8 GGUF (template strings are
quant-independent), llama-server b10066, real
`Orchestrator → Runner → LlamaCppAdapter` path — no marvin involvement:

| Cell | Fixed template (v22.5) |
|---|---|
| capital of France, default effort | `Paris`, `stop`, prompt 24 tokens |
| `2+2`, `enable_thinking: false` | `4`, `stop`, 0 reasoning |
| single-turn `get_weather` tool call | correct call, `finish tool_calls`, 129 reasoning chars |
| JSON-string args + `sunny, 21C` tool result | synthesizes `sunny 21°C`, `stop` (native re-emits the call) |

Poison control: a template containing only `raise_exception(...)` fails
the start with `llama-server exited with 1 … the supplied chat template is
not supported` — the file on disk is the one rendered.

Same-shape check on marvin's NVFP4 checkpoint (native vs fixed sidecar):
plain-turn prompt 66 → 24 tokens; JSON-string history fixed as above;
`enable_thinking: false` accepted on both (the hard-crash report targets a
different checkpoint shape, not giq's).

## Consequences

- An operator runs a fixed template with one recipe line:
  `params: {chat_template_file: qwen-fixed.jinja}` (file under
  `GIQ_MODELS_DIR`). `GET /storage` and strict validation apply as for any
  parameter; a leading-dash value is refused (argv injection).
- A template change is a behaviour change: re-measure the recipe's VRAM
  figure and confirm DRY/budget settings against the new system prompt.
- Supply chain: a single-maintainer file with full prompt control. Pin the
  file content (sha) alongside the recipe, as with weight revisions.
- Sticky inline tags (`<|think_off|>` etc.) let user content steer
  reasoning — acceptable for a single-operator homelab, a policy question
  for multi-user service before any fleet default.

## Alternatives considered

- **Per-request `chat_template` forwarding.** Rejected: needs the server
  trust flag, smuggles deployment config into the client contract, and giq
  drops the field today — silently honouring it would change behaviour for
  every existing client.
- **Fleet default now.** Rejected: evidence supports opt-in, not a flip.
  The numbers behind the current recipes were measured against the native
  template.
- **Patching stock behaviour with budgets/guards only.** Already done at
  other layers and kept; the template fixes a different layer (rendering
  crashes, default injection) that samplers cannot reach.
