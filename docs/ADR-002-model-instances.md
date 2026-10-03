<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# ADR-002: Model instances — weights × engine × parameters

**Status:** Accepted, amended by [ADR-003](ADR-003-domain.md)  
**Date:** 2026-09-28  
**Authors:** giq maintainers

> **Amended by [ADR-003](ADR-003-domain.md).** What this record calls an
> *instance* — the YAML file — is a **recipe** now; *instance* names a
> recipe running on a card. Files are `<name>.yaml` under `giq/recipes/`
> and `GIQ_RECIPES_DIR`, keyed by a globally unique name with a
> `modalities` list rather than `(worker, name)`, and "worker" is
> **modality**. The decisions on files, schemas, VRAM and the adapter
> contract stand as written below.

## Context

Everything giq knows about a model is compiled into the code today:

- `giq.registry.ModelSpec` — worker, name, VRAM figure, residency priority,
  vision/mmproj, aliases, lane width.
- About ten per-model tables in `giq/workers/llm.py` keyed by model name —
  `MODEL_PATHS`, `MODEL_CTX_SIZE`, `MODEL_CACHE_TYPE_K/V`, `MODEL_REASONING`,
  `MODEL_REASONING_BUDGET`, `MODEL_REQUEST_DEFAULTS`, `MODEL_SPEC_TYPE`,
  `MODEL_PARALLEL`, `MODEL_LOOP_GUARD`, `MODEL_ALIAS`.
- `config.yaml` — image model files (`image_models`), engine binaries,
  residents.
- Environment variables for the transformers children (OCR, depth, multiview).
- The `model_policy` table — runtime residency policy and GPU binding.

That worked while one person ran one machine. It no longer does:

1. **Adding a model means changing giq.** An organisation that runs giq
   with its own models — research variants, fine-tunes, vendor checkpoints —
   has to fork the registry. Models that must not be in the public catalog
   have nowhere to live.
2. **One file, several configurations.** The same weights already serve as
   two models (a vision profile with mmproj and a fast profile with
   speculative decoding, which llama.cpp cannot combine). Each variant costs
   an entry in every table, and a test keeps the tables from drifting.
3. **More than one engine per modality.** Vendor checkpoints (NVIDIA
   ModelOpt NVFP4/FP8) run natively only on engines such as vLLM. For
   multi-user serving, vLLM's continuous batching beats llama-server's fixed
   slots. Choosing the engine becomes a property of a deployment, not of the
   worker type.
4. **No way to try a setting without a code change.** Context size, KV cache
   type, speculative decoding or a VRAM budget can only be changed by editing
   Python and restarting.

## Decision

Introduce three concepts and make the **instance** the unit that clients
call, the queue schedules and the dashboard edits.

| Concept | What it is | Owned by |
|---|---|---|
| **Weights** | Where the model's files are and what they are: local path or Hugging Face repo + revision, format (`gguf`, `safetensors`, `modelopt`), modality, projector file, licence | instance file |
| **Engine** | A runtime that can serve weights: binary or interpreter, version probe, the formats it accepts, and a **parameter schema** | code (adapter) + `config.yaml` (binary path) |
| **Instance** | The name clients send (`model`), one weights reference, one engine, that engine's parameter values, residency defaults, and the measured VRAM | YAML file |

An instance is written `name@engine` in logs and the dashboard
(`qwen3.8-27b@llama.cpp`, `qwen3.8-27b-nvfp4@vllm`), but clients keep
sending the plain `name` — the engine never leaks into the API contract.

### D1: Instances are YAML files

One file per instance under an instances directory:

- built-in instances ship in the package (`giq/instances/*.yaml`) and
  replace today's registry and per-model tables;
- operator instances live in `GIQ_INSTANCES_DIR` (default
  `~/.config/giq/instances`), are loaded after the built-ins, and may
  override a built-in by using the same name.

Files, not the database, are the source of truth because a deployment wants
to **review, version and reproduce** its model setup: an organisation keeps
its instances directory in its own repository, diffs changes in review, and
brings up an identical second machine by copying it. Models that must not
be public — internal fine-tunes, research variants — never touch the giq
repository.

Runtime state stays where it is: residency policy overrides and GPU
bindings made from the dashboard remain rows in `model_policy`. They are
operator decisions about the running machine, not model configuration.

```yaml
# ~/.config/giq/instances/qwen3.8-27b-nvfp4.yaml
name: qwen3.8-27b-nvfp4
label: Qwen3.8 27B (NVIDIA NVFP4)
worker: llm
engine: vllm
weights:
  source: hf:nvidia/Qwen3.8-27B-NVFP4
  revision: 3f1c0de          # pinned; a moving revision re-invalidates the measurement
  format: modelopt
  licence: apache-2.0
capabilities: [chat, tools, vision]
params:                      # validated against the vllm adapter's schema
  gpu_memory_utilization: 0.72
  max_model_len: 131072
  max_num_seqs: 16
  enforce_eager: false
residency:
  default_policy: pinned     # vllm cold starts take minutes: never on-demand
  gpu: auto
vram:
  gb: 23.0
  measured: true
  measured_on: NVIDIA GeForce RTX 5090
  measured_with: { engine_version: "0.30.0", params_hash: 9b2e41 }
request_defaults:
  temperature: 0.6
```

### D2: Engines declare a parameter schema

Each engine adapter declares its parameters once — name, type, default,
bounds or allowed values, whether a change needs a reload, and a help text
key for the dashboard's translations. Instance files are validated against
it on load and on save; an unknown or out-of-range parameter is an error,
never silently ignored.

The dashboard renders the instance editor from the schema, so a new engine
needs an adapter, not new UI code. Examples:

| Engine | Parameters (excerpt) |
|---|---|
| llama.cpp | `ctx_size`, `cache_type_k/v`, `flash_attn`, `spec_type`, `mmproj`, `chat_template_file`, `reasoning`, `reasoning_budget`, `parallel`, `loop_guard`, `ready_timeout` |
| vllm | `gpu_memory_utilization` or `kv_cache_memory_bytes` (**one required**), `max_model_len`, `max_num_seqs`, `enforce_eager`, `chat_template_file`, `sleep_mode` |
| sd.cpp | `offload_to_cpu`, `vae_tiling`, `steps`, `cfg_scale`, `sampler`, `scheduler` |
| transformers | `dtype`, `max_pages`, `pages_per_pass`, … per worker |

There is deliberately **no free-form `extra_args`**: an instance ultimately
becomes a command line, and a string passed through to it is a remote code
execution vector once the dashboard can edit instances. A flag an operator
needs becomes a schema parameter.

### D3: VRAM stays measured

giq's VRAM gate is only as good as its figures, and the rule that figures
are measured, not derived from file size, carries over:

- a new or edited instance whose weights, engine version or VRAM-relevant
  parameters changed is **unmeasured**;
- an unmeasured instance loads only if it declares an upper bound
  (`vram.gb` with `measured: false`), which the gate uses as-is;
- **Measure** (API and dashboard) loads the instance on a chosen card with
  nothing else of giq's on it, runs a probe request at the configured
  context, records the peak and writes `measured: true` with what it was
  measured with;
- engines that reserve memory up front (vllm) must be given a budget in
  their parameters; the measurement then confirms the budget holds instead
  of discovering it.

### D4: One engine adapter contract

An engine adapter turns (weights, params) into a running server and back:

- `build_command(instance, card) -> argv` — from schema-validated params
  only;
- `health()`, `ready()`, `stop()` — the lifecycle the runner already drives
  for llama-server;
- `endpoint()` — engines that speak the OpenAI protocol (llama.cpp, vllm)
  are proxied by `openai_compat` unchanged;
- `accepts(format)` — the loader refuses an instance whose weights format
  the engine cannot read (`modelopt` on llama.cpp, `gguf` on vllm) at load
  time, not at the first request.

Engines keep their own interpreters where their dependencies clash with
giq's (`envs/<engine>/`, as `envs/da3` and `envs/unlimited-ocr` do today).

### D5: Editing from the dashboard

The Models view gets an instance editor: create from weights, **Duplicate
as…** from an existing instance, edit parameters (form rendered from the
schema, reload-requiring changes marked), Measure, and delete. Saving writes
the YAML file in `GIQ_INSTANCES_DIR`; built-in instances are read-only and
are customised by duplicating or overriding them.

Editing is an administrative action: when giq is reachable beyond loopback
it requires the access token, and the file write is atomic (write + rename)
so a crash never leaves a half-written instance.

### D6: Benchmark next to Measure

Choosing between instances of the same weights — a quantisation, a second
engine, a speculative-decoding profile — needs numbers from the serving path
itself, not from a separate benchmark tool with different flags. **Benchmark**
(API and dashboard) runs a fixed workload through giq's own API against an
instance: load time, time to first token, generation tok/s at 1, 4 and 16
concurrent requests, and a small fixed prompt set in each UI language whose
answers are kept for side-by-side reading. Results are stored per instance
together with the engine version and parameter hash they were taken with,
and the Models view compares instances side by side. It runs through the
queue like any other job, so it needs no pause and is subject to the same
eviction rules as production traffic.

### D7: Live reload

Instances change without restarting giq, by an explicit command —
`POST /control/reload`, the dashboard's Reload action, or saving in the
instance editor — never by a file watcher, so a half-edited file never goes
live.

- **Validate, then swap.** The whole instances directory is parsed and
  validated against the engine schemas into a new immutable snapshot, which
  replaces the current one in a single reference swap. Any error leaves the
  running snapshot untouched and is reported by the reload.
- **Diff against what is running:**
  - added → available immediately;
  - changed, not loaded → takes effect on the next load;
  - changed and loaded → marked stale, keeps serving with its old settings,
    restarts with the new ones when idle — a running job is never cut;
  - removed → unloaded when idle; queued jobs for it fail with a clear error;
  - parameters that need no reload (request defaults, labels, residency
    policy) apply immediately.
- The runner reads specs from the current snapshot at every decision point
  instead of from module constants — the same refactor migration step 1
  needs.

### D8: Concurrency is derived, not configured

How many requests an instance serves at once is a property of its engine
parameters, and giq takes it from there instead of keeping a number of its
own. Today the two drift: the registry's lane width says 4 while llama-server
runs with one slot, so giq hands over four requests that are then served one
after another, and the dashboard advertises parallelism that does not exist.

- The engine parameter lives in the instance — `parallel` (and `kv_unified`)
  for llama.cpp, `max_num_seqs` for vllm, a fixed 1 for the transformers
  children, which batch inside a job instead.
- The adapter derives what giq needs:
  `concurrency(params) -> (max_parallel, per_request_context, shared_kv)`.
  The scheduler dispatches at most `max_parallel` requests to the instance,
  and the dashboard shows exactly that ("up to 16 in parallel · 180k tokens
  shared" or "up to 4 in parallel · 32k context each"). There is no separate
  lane width to set.
- The parameters mean different things per engine, and the schema says so:
  llama.cpp without `kv_unified` splits `ctx_size` evenly across its slots,
  so `parallel: 4` at 131072 leaves 32768 per request — the editor shows the
  effective context per request and warns below a threshold. vllm's
  `max_num_seqs` is only a ceiling over a paged KV pool that any mix of
  requests shares.
- Concurrency parameters change the KV cache, so changing them makes the
  instance unmeasured (D3); Benchmark (D6) records throughput at 1, 4 and 16
  parallel requests, which is where the difference between engines shows —
  measured on an RTX 5090 with Qwen3.8-27B: 56 tokens/s in total at any load
  through a one-slot llama-server, 937 tokens/s at 16 parallel requests
  through vllm.

### D9: Engine knowledge in code, choices in instances — with profiles

The line between code and instance files is: giq knows *how*, the instance
decides *whether*.

**In the engine adapters (code):**

- **How a feature is switched on.** Instances name features abstractly —
  `speculative: {method: mtp, tokens: 2}`, `vision: true`,
  `kv_cache: fp8` — and the adapter turns them into its engine's flags
  (`--spec-type draft-mtp` for llama.cpp, `--speculative-config` for vllm).
- **What the weights can do.** The adapter inspects the checkpoint — an MTP
  head (`blk.N.nextn.*` in a GGUF, `mtp_num_hidden_layers` in a Hugging Face
  config), a vision tower, the trained context — and the editor offers only
  what is there.
- **Which parameters conflict.** Schema rules, not documentation: llama.cpp
  cannot combine speculative decoding with a projector, so vision and MTP
  are two instances of one file; vllm's MTP layer needs KV cache of its own,
  so the budget and context are re-checked when it is switched on;
  llama.cpp's `parallel` divides the context (D8).
- **How to start the engine safely.** The process environment and limits
  that every instance of an engine shares: toolchain pins in its `envs/`
  project, compiler and cache locations under `GIQ_HOME/cache`, a memory
  ceiling on the engine process, and a one-off *prepare* step for kernels an
  engine compiles on first use — FlashInfer's sm_120 GEMMs take minutes and
  more than 20 GB of RAM to build, which must never happen inside a request
  or unbounded on a shared machine.
- **How to measure it.** Measure and Benchmark (D3, D6) read the engine's own
  metrics where it has them — vllm reports speculative acceptance per draft
  position.

**In the instance (YAML or the editor):** the weights, the engine, and the
choices — speculative decoding and how far, context, parallelism, KV cache
format, memory budget, residency, GPU binding — plus the figures Measure and
Benchmark write back.

**Profiles** keep the choices simple. Each adapter ships named parameter
sets derived from measurements; an instance names one and overrides
individual values:

| Profile | llama.cpp | vllm |
|---|---|---|
| `interactive` | MTP draft, no projector, one slot | MTP, 2 draft tokens, up to 4 parallel, larger batch budget |
| `throughput` | no speculation, 4 slots with unified KV | no speculation, up to 16 parallel |
| `vision` | projector, no speculation | vision tower on |

The profile defaults are the measured trade-off, not a preference. On an RTX
5090 with NVIDIA's NVFP4 checkpoint of Qwen3.8-27B under vllm, MTP with three
draft tokens raised one request from 72 to 117 tokens/s but lowered the total
at 16 parallel requests from 937 to 750, and accepted drafts fell to 0.66,
0.37 and 0.17 by position — so interactive uses MTP with two draft tokens and
throughput uses none.

```yaml
name: qwen3.8-27b-nvfp4-chat
engine: vllm
weights: {source: hf:nvidia/Qwen3.8-27B-NVFP4, format: modelopt}
profile: interactive
params:
  max_model_len: 131072
```

## Consequences

- The registry and the per-model tables in `llm.py` shrink to the built-in
  instance files; their defaults move into the engine schemas. The tables'
  documented reasoning (why a context size, why a KV type) moves into
  comments in the YAML files.
- Model-specific tests become schema and loader tests plus fixture
  instances; the "profiles over one file must not drift" invariant becomes a
  property of **Duplicate as…** instead of a test over parallel tables.
- An organisation's private models never enter the giq repository.
- A second LLM engine (vllm) becomes an adapter plus a schema, and the
  question "which engine is faster for us" is answered by two instances of
  the same weights side by side.
- Cost: a loader, schema validation, an adapter layer around what the
  runner does inline today, and the editor UI. The migration is mechanical
  but touches the most tested code in giq.

## Migration

1. **Data model and loader, no behaviour change.** Define the instance
   schema, generate `giq/instances/*.yaml` from today's registry and tables,
   load them into the existing `ModelSpec`/table shapes. The full test suite
   must pass unchanged — that is the proof nothing moved.
2. **Engine adapters and schemas.** llama.cpp, sd.cpp and the transformers
   children behind the adapter contract; the tables disappear.
3. **API.** List, get, validate, save, duplicate, delete, measure,
   benchmark, reload.
4. **Dashboard.** Instance editor, Measure/Benchmark and Reload in the
   Models view.
5. **vllm adapter** (`envs/vllm`) as a second LLM engine, gated on a
   benchmark against llama.cpp on the same weights: load time, VRAM,
   quality, and throughput at 1, 4 and 16 concurrent requests.

## Alternatives considered

- **Database as the source of truth, edited only in the dashboard.** Simpler
  to write from the UI, but not reviewable, not diffable, and a second
  machine is a database copy instead of a directory. Rejected; runtime
  policy stays in the database where it already is.
- **Keep the code registry, add an override file.** Solves private models
  but keeps two ways to define one, and still needs a code change for every
  new engine parameter.
- **Free-form engine arguments per instance.** Maximum flexibility, and a
  command-injection path from the dashboard. Rejected (D2).

## Open questions

- Does the instance name stay unique across workers, or is it
  `(worker, name)` as `ModelSpec.key` is today?
- Weights on Hugging Face: download on first load, or only via an explicit
  "fetch" action (disk space, licences, offline machines)?
- Should Measure be allowed to evict residents, or only run when the chosen
  card is free?
- Per-instance access control (which tokens may call which instances) —
  here or in a later ADR on multi-tenant access?
