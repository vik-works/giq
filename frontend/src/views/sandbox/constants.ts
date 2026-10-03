// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

/* A thinking model spends its token budget on the thought before it writes a
   word of the answer, so a budget sized for the answer runs out mid-thought
   and llama-server returns content_len=0 — an EMPTY answer, not a short one,
   and no error to explain it (measured on qwen3.6-27b with a budget of 500).

   On qwen3.6-27b, "explain in two sentences why the sky is blue" with
   thinking on spent all 512 tokens to produce 304 characters — ~450 tokens of
   invisible thought on a trivial question. A real prompt does not fit.
   max_tokens is a ceiling, not an allocation, so raising it costs nothing
   when the thought is short.

   8192 was still fitted to easy questions. Measured on qwen3.8-27b with real
   problems: a race-condition diagnosis spent 20,284 tokens, a design question
   21,009, an underspecified puzzle 10,827 — 8192 would have truncated three
   of four into empty answers. 32768 is Qwen's own recommendation for thinking
   mode and covers all of them. It is only reachable because the answer
   streams: nine minutes of generation outlives every timeout a single
   response has to survive. */
export const THINKING_TOKEN_FLOOR = 32768;

/* The tool-calling panel runs one model on purpose: tool calling depends on
   the chat template's tool grammar, and gemma-4-12b is the registry's small
   default resident whose template is known to emit OpenAI-style tool_calls.
   Its served contract is thinking-on by default and this panel never opts
   out, so it gets the same THINKING_TOKEN_FLOOR budget as the chat panel —
   512 was a budget the model could spend entirely on the thought. */
export const TOOLS_MODEL = "gemma-4-12b";

/** The ask → execute → answer loop gives up after this many model turns. */
export const TOOLS_MAX_ROUNDS = 4;
