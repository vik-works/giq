<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# Configuration

`config.yaml` in the working directory (or `GIQ_CONFIG`) holds engines,
GPU placement, residents, access and the data directories. The
checked-in file is annotated; adjust its paths to your system:

```yaml
engines:
  llama.cpp: /path/to/llama-server
  sd.cpp: /path/to/sd-server
```

What each model is — its weights, engine, parameters and VRAM figure — is not
in `config.yaml` but in its [recipe file](#recipes).

Engines are described in [engines.md](engines.md); `access:` in
[access-and-privacy.md](access-and-privacy.md).

## Where giq keeps its files

Set `GIQ_HOME` and everything that is not code lives under one directory —
`python -m giq init` creates it with a commented `config.yaml`:

```
$GIQ_HOME/
  config.yaml   the one config
  models/       weights
  recipes/      your recipe files (ADR-002, ADR-003)
  engines/      engine builds, e.g. engines/llama.cpp/bin/llama-server
  state/        stats.db, in-flight log
  cache/        Hugging Face and kernel caches
```

Each location resolves as: its environment variable, else the `paths:` block
in `config.yaml` (absolute, or relative to `GIQ_HOME`), else the `GIQ_HOME`
layout, else the defaults below. Without `GIQ_HOME` nothing moves: models in
`~/models`, state in the checkout's `data/`, `./config.yaml`. `GET /storage`
reports the resolved paths. A server install is described in
[deployment.md](deployment.md).

Environment variables:

| Variable | Default | What |
|----------|---------|------|
| `GIQ_HOME` | unset | Data root (layout above) |
| `GIQ_HOST`, `GIQ_PORT` | `127.0.0.1`, `8084` | Bind address (command-line flags win) |
| `GIQ_DATA_DIR` | `$GIQ_HOME/state`, else `data/` | Stats database and in-flight log |
| `GIQ_RECIPES_DIR` | `$GIQ_HOME/recipes`, else `~/.config/giq/recipes` | Recipe files |
| `GIQ_ENGINES_DIR` | `$GIQ_HOME/engines` | Engine builds, looked up before PATH |
| `GIQ_CACHE_DIR` | `$GIQ_HOME/cache` | Caches, exported to the engine children as `HF_HOME`, `XDG_CACHE_HOME` and friends |
| `GIQ_MODELS_DIR` | `$GIQ_HOME/models`, else `~/models` | Root of the model store; relative weight paths in recipe files resolve against it |
| `GIQ_CONFIG` | `$GIQ_HOME/config.yaml`, else `./config.yaml` | Config file |
| `GIQ_LLAMA_BINARY` | `llama-server` on PATH | llama.cpp server binary |
| `GIQ_SDCPP_BINARY` | `sd-server` on PATH | stable-diffusion.cpp server binary |
| `GIQ_VLLM_PYTHON` | `envs/vllm/.venv/bin/python` | The vllm engine's interpreter; `vllm serve` is the console script beside it ([engines.md](engines.md#vllm)) |
| `GIQ_OCR_MODEL_DIR`, `GIQ_GLM_OCR_MODEL_DIR`, `GIQ_GLM_LAYOUT_DIR` | the recipe's | The `unlimited-ocr` snapshot, the `glm-ocr` snapshot and its layout part; each outranks that built-in's `weights` (see [Weights](#weights)) |
| `GIQ_DEPTH_MODELS_DIR` | `GIQ_MODELS_DIR` | Root for the relative weight paths of depth recipes |
| `GIQ_AUDIO_WHISPER_MODEL`, `GIQ_AUDIO_DIAR_MODEL`, `GIQ_EMBED_MODEL` | the recipe's | Repository or path the audio and voiceprint children load, outranking the recipe's `weights` |
| `GIQ_GPU_DEVICE` | biggest card | Default GPU, index or NVML UUID |
| `GIQ_TOKEN` | unset | Shared access token (see [Access](access-and-privacy.md#access)) |
| `GIQ_STATS_DB` | `<data dir>/stats.db` | Stats database |
| `GIQ_INFLIGHT_LOG` | `<data dir>/inflight.log` | In-flight job log (job shapes only, see [Privacy](access-and-privacy.md#privacy)) |
| `GIQ_OCR_MAX_UPLOAD_MB`, `GIQ_OCR_MAX_PAGES` | 64, 200 | Upload limits (see [OCR](api.md#ocr)) |

## Recipes

Every model giq serves is a **recipe**: one YAML file naming it, the
modalities it serves, its weights, the engine that runs them, that engine's
parameters, residency defaults and the VRAM figure the scheduler gates on
([ADR-002](ADR-002-model-instances.md); the terms are
[ADR-003](ADR-003-domain.md)'s). A running recipe is an *instance*. Two
places hold recipes:

- **Built-in** — one file per recipe shipped in the package,
  `src/giq/recipes/<name>.yaml`. Read them for the catalog giq
  ships and for why each model runs with the settings it does; the reasoning
  is in their comments. Don't edit them in an installed giq.
- **Yours** — `*.yaml` / `*.yml` directly in the recipes directory
  (`GIQ_RECIPES_DIR`, `paths.recipes`, `$GIQ_HOME/recipes`, else
  `~/.config/giq/recipes`). File names are free; the contents say what the
  file defines.

A recipe is identified by its `name`, which is what a client sends as
`model`; names and aliases are unique across modalities. A recipe lists the
`modalities` it serves — usually one; `flux_klein` serves `text2image` and
`image_edit` from one sd-server, so alternating renders and edits costs no
reload. A file of yours with the same name as a built-in **replaces** it
entirely — nothing is inherited, so parameters you leave out take the
engine's defaults, not the built-in's; copy the built-in file and change what
you need. A file with a new name **adds** a recipe. The log says at INFO which
file each of your recipes came from and which built-ins they replace. A file
written before ADR-003 with `worker: llm` still loads, as
`modalities: [llm]`, with a warning.

```yaml
# ~/.config/giq/recipes/qwen3.8-27b.yaml — the built-in at 196k context
name: qwen3.8-27b
modalities: [llm]
engine: llama.cpp
detail: "chat + vision · 192k ctx"
weights:
  path: unsloth-Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf   # under GIQ_MODELS_DIR
  source: hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
  revision: 4ca720788d1e01f1bff70c033e0d0028fd02e502
  format: gguf
  parts:
    mmproj: unsloth-Qwen3.8-27B-GGUF/mmproj-F16.gguf      # the vision projector
capabilities: [chat, vision]
params:
  ctx_size: 196608
  cache_type_k: q8_0
  cache_type_v: q8_0
  reasoning: "on"            # quoted: a bare on is YAML's true
vram:
  gb: 29.0
  measured: false            # an upper-bound estimate until measured
max_batch: 32
```

| Key | What |
|-----|------|
| `name` | What clients send as `model`; unique across modalities. Letters, digits, `.`, `_`, `-`. |
| `modalities` | What it serves, one or more of `llm`, `text2image`, `image_edit`, `tts`, `stt`, `audio`, `embed`, `ocr`, `depth` |
| `engine` | The runtime, which must be able to serve every listed modality: `llama.cpp` or `vllm` (llm); `sd.cpp` (both image modalities); `kokoro`, `faster-whisper`, `faster-whisper+pyannote`, `speechbrain`, `transformers` for the rest |
| `label`, `detail` | Dashboard presentation; `label` defaults to the name |
| `weights.path` | The weights file (a checkpoint directory for `vllm`), relative to `GIQ_MODELS_DIR`; `~` or absolute is used as written. Required for `llama.cpp` and `vllm`; `vllm` also needs `format: safetensors` or `modelopt` |
| `weights.parts` | The other files the model needs, by the name its modality's engine reads them under (see [Weights](#weights)) |
| `weights.source`, `revision` | Where the weights come from: `hf:org/repo` for a whole repository, `hf:org/repo/path/in/repo` for one file of it (which then needs a `path` to go to), at a pinned commit |
| `weights.format`, `licence` | `gguf`/`safetensors`/`modelopt`, and the licence the weights are under |
| `capabilities` | `chat`, `vision`. For llama.cpp `vision` needs `weights.parts.mmproj` and vice versa; for vllm, leaving it out serves a multimodal checkpoint as text |
| `profile` | vllm only: a named parameter set (`interactive`, `throughput`) under `params`, which outrank it ([engines.md](engines.md#profiles)) |
| `params` | The engine's parameters, below. Unset ones take the engine's defaults |
| `request_defaults` | Body fields sent under each request; the caller's own values win. llama.cpp: samplers, `reasoning_budget_tokens`; vllm: `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` |
| `residency.priority` | Position in the default resident set, lowest first; unset = loads on demand |
| `vram.gb`, `vram.measured` | The gate's figure, and whether it was observed on real hardware (`measured_on`, `notes` optional). vllm with a `kv_cache_memory` budget: `vram.weights_gb` + `vram.overhead_gb` instead, and `gb` is their sum with the budget |
| `aliases` | Other names clients may send; unique like names |
| `lane_width`, `max_batch`, `voices` | Concurrent jobs on a resident's lane (default per modality; for vllm it is `params.max_num_seqs` and cannot be set), batch ceiling (a number, or one per modality), TTS voices |

llama.cpp `params`: `ctx_size` (shared by the slots), `parallel` (slots;
more than one turns on continuous batching), `cache_type_k` and
`cache_type_v` (set together and equal — a mixed pair falls off the fused
attention kernel), `reasoning` (`on`, `off`, `auto`, or `template` to let the
chat template decide), `reasoning_budget`, `spec_type`, `alias`,
`loop_guard`, `ready_timeout` (seconds a start may take, default 300 — raise
it for a big GGUF on a slow disk; a server that exits while starting fails at
once, and its output is in `<state>/logs/llama-<model>.log`). vllm `params` — a VRAM budget (`kv_cache_memory`, preferred,
or `gpu_memory_utilization`; exactly one), `max_model_len` (required),
`max_num_seqs`, `max_num_batched_tokens`, `kv_cache_dtype`, `speculative`,
`enforce_eager`, `reasoning_parser`, `tool_call_parser`, `memory_max`,
`ready_timeout` — are described in [engines.md](engines.md#parameters). The
other engines take no parameters from a recipe yet.

### Weights

Every engine loads the weights its recipe names, so a recipe file with a
new name is a new model — no table in giq's code has to know it. The main
weights are `weights.path`; the other files a model needs are
`weights.parts`, each a path or a mapping with its own source:

```yaml
weights:
  path: zai-GLM-OCR                      # under GIQ_MODELS_DIR
  source: hf:zai-org/GLM-OCR
  revision: ca5d8b3e287e52589e37c28385d9655ee4372f9d
  parts:
    layout:                              # a mapping, with its own source
      path: PaddlePaddle-PP-DocLayoutV3
      source: hf:PaddlePaddle/PP-DocLayoutV3_safetensors
      revision: 97d101e6db2642e162a1d05392d1b0231c91033e
```

A source names a whole repository, whose files go in the directory `path`
names, or one file of it, which becomes the file `path` names: one
quantisation of a GGUF repository, or one component of a ComfyUI-style
repository (`hf:Comfy-Org/z_image_turbo/split_files/vae/ae.safetensors`).
Without a `path`, a whole repository is loaded from the Hugging Face cache.

| Modality | Main weights | Parts |
|--------|--------------|-------|
| `llm` | the GGUF | `mmproj` (llama.cpp's vision projector; `params.mmproj` in an older file is read as this part, with a warning) |
| `text2image`, `image_edit` | — | `diffusion`, `text_encoder`, `vae` (required), `lora` |
| `ocr` | the snapshot directory | `layout` (engine `transformers`: GLM-OCR's layout model) |
| `depth` | the snapshot directory | — |
| `stt` | a CTranslate2 directory, else the `hf:` source | — |
| `audio` | — | `asr`, `diarization` (`hf:` sources) |
| `embed`, `tts` | the `hf:` source | — |

A part the modality's adapter does not read is refused, like an unknown key. The OCR child
is chosen by the checkpoint: the architecture its `config.json` names
(`UnlimitedOCRForCausalLM` runs Unlimited-OCR's pipeline,
`GlmOcrForConditionalGeneration` GLM-OCR's behind its layout model), so an
OCR recipe of your own is another checkpoint of one of the two. Both run on
the `transformers` engine; `transformers-4.57`, Unlimited-OCR's former
interpreter, is read as `transformers` with a warning.

The environment variables that located these snapshots before recipe
files existed still work, and outrank the file: `GIQ_OCR_MODEL_DIR`,
`GIQ_GLM_OCR_MODEL_DIR` and `GIQ_GLM_LAYOUT_DIR` replace the paths of the
built-in `unlimited-ocr` and `glm-ocr` (and only theirs — an OCR recipe
under another name is not redirected), `GIQ_DEPTH_MODELS_DIR`
replaces the models directory for depth recipes' relative paths, and `GIQ_AUDIO_WHISPER_MODEL`, `GIQ_AUDIO_DIAR_MODEL` and
`GIQ_EMBED_MODEL` replace what the audio and voiceprint children load.
`GET /weights` lists every checkpoint where giq resolved it, and whether it
is there.

### Getting a recipe's weights

Each recipe is, on this machine, `ready` (it runs), `fetchable` (its
weights are missing and giq can fetch them), `manual` (missing, and the
recipe says nowhere to fetch them from: place the files where `weights.path`
says) or `unfit` (it cannot run here: too large for every card, a card its
engine cannot use, or the engine's binary is missing). `GET /recipes` says
which, with the reasons; `/capabilities` and `/v1/models` offer only the
ready ones.

```bash
giq add glm-ocr --dry-run   # what it takes: size, access, licence, disk
giq add glm-ocr             # the same, then fetch (asks once; -y does not)
```

Before fetching, giq plans: whether a card fits it, the download size from
the Hugging Face Hub, free disk where the files go, parts another recipe
already brought, whether a gated repository opens with your token, and the
licence. `GET /recipes/{name}/plan` returns the same plan. A fetch that
stops (Ctrl-C, a lost connection) resumes when run again; files appear in
place only once every one of them is there.

The dashboard and `POST /recipes/{name}/fetch` fetch from the service
itself, when it may write where the files go. Under the hardened systemd
unit the models directory is read-only to the service: run `giq add` as the
operator, or add the models directory to the unit's `ReadWritePaths`.

Gated repositories (pyannote's diarization model) need a Hugging Face token
of an account that accepted their terms: `HF_TOKEN` in the environment, or
`hf auth login` as the user giq runs as. giq never stores or shows it.

### Image models

An image recipe's files are its `weights.parts`. The built-ins
expect them under the models directory, one subfolder per part —
`diffusion_models/`, `text_encoders/`, `vae/`, `loras/`. To keep them
elsewhere, override the recipe with absolute paths:

```yaml
# ~/.config/giq/recipes/zimage.yaml
name: zimage
modalities: [text2image]
engine: sd.cpp
weights:
  parts:
    diffusion: /srv/image-models/diffusion_models/z_image_turbo_bf16.safetensors
    text_encoder: /srv/image-models/text_encoders/qwen_3_4b.safetensors
    vae: /srv/image-models/vae/ae.safetensors
vram:
  gb: 13.0
  measured: true
max_batch: 8
```

`max_batch` may differ per modality: `flux_klein` takes
`{text2image: 8, image_edit: 4}`.

**`image_models` in `config.yaml` is no longer read.** An image recipe's
files, engine and VRAM figure are its own; giq logs a warning when the block
is still there. The old engine spelling `sdcpp` is read as `sd.cpp`, with a
warning.

**Validation is strict.** An unknown key, a key given twice, a parameter the
engine does not have, a modality paired with an engine that cannot serve it,
and a setting giq does not honour yet (`profile` on an engine without
profiles, `residency.gpu`, `residency.default_policy: off`) are all errors, never silently ignored. A
file of yours that fails is logged as an error and left out — the built-in of
that name, if there is one, keeps serving — and two of your files defining
the same recipe are both left out, since which one won would be an accident
of sorting. Recipe files are read at startup; restart giq after changing
them. `GET /storage` lists the files that were left out with the reason
(its `recipes` block), and the dashboard's Recipes view shows them as a
warning above the catalog.

## Residency

Every recipe has a residency policy, set in the dashboard's Recipes view or
with `PUT /recipes/{name}/residency` (`{"policy": …}`; `DELETE` returns it
to the default). A recipe giq keeps loaded is a *resident* instance; one it
loads for a job is *on demand*:

- `pinned` (**keep warm**) — kept loaded whenever VRAM allows, reloaded after
  an eviction and on boot.
- `auto` (**on demand**, the default) — loads when a job arrives, is evictable,
  and unloads after two idle minutes.
- `off` — refuses jobs and cannot load by any path.

**Nothing is kept warm out of the box:** no built-in recipe sets a
`residency.priority`, so a new install loads a model only when a job asks for
it. Keep models warm by listing them under `residents:` in `config.yaml`, in
reload-priority order, or with the dashboard. Overrides persist in `stats.db`
and outrank `config.yaml`'s `residents:`, which outranks a recipe's own
`residency.priority` (set it in a recipe file of yours).
Before a load, giq gates on the recipe's declared VRAM figure (measured on
real hardware where the recipe says so) plus a margin against the card's
free VRAM, and evicts keep-warm instances on that card when that is what it
takes. `GET /instances` lists what is running, card by card.

## Multiple GPUs

Recipes that name no card run on the **default** card — the biggest one unless
you say otherwise, by index or NVML UUID:

```yaml
gpu:
  device: GPU-d0fda82c-452e-bbf7-561b-07d3a056ecfd   # or "1", or GIQ_GPU_DEVICE
```

**Bind a recipe to a card** and it is gated against that card's VRAM, loads
there, gets that card's server port, and can only ever evict residents that
share it — so a render on one card cannot cost you the LLM on the other:

```bash
curl -X PUT localhost:8084/recipes/flux_klein/card \
  -H 'content-type: application/json' -d '{"device": 1}'

curl -X PUT localhost:8084/recipes/flux_klein/card \
  -H 'content-type: application/json' -d '{"device": null}'   # unbind
```

Bindings take an index or a UUID and are stored as the UUID; they persist in
`stats.db`, survive restarts and re-enumeration, and are independent of
residency (pinning does not bind, unbinding does not unpin). The dashboard
does the same thing visually: each GPU card lists what is loaded on it, the
residency budget is drawn per card, and each recipe card's GPU picker binds
it with one click. A checked-in default
lives under `gpu.bind` in `config.yaml`, and `gpu.reserve` leaves headroom on
a card shared with a desktop.

Every VRAM figure in the API describes one card. `/status`'s `vram_*`
scalars are the default card's, named in `/status.gpu`; `/status.gpus` lists
every card in the same terms, and a job waiting for room is judged on the
card it is bound to (`vram_blocked_gpu`). Each recipe's `card` in
`/recipes` names its own, and `/gpus` adds temperature, power and
the per-process split.

Used VRAM is reported split two ways — `vram_giq_gb` (instances giq is holding,
with a per-process `giq[]` breakdown) and `vram_other_gb` (the desktop, other
CUDA apps, a game). Only the first is something giq can free by unloading, so
a full card and a full card *giq caused* are shown as different things. The
dashboard draws it as a two-segment gauge: ours solid, theirs hatched.

Instances are spawned with `CUDA_VISIBLE_DEVICES` set to their card. This matters
most for llama.cpp, whose default `-sm layer` otherwise spreads a model's
layers *and KV cache* across every visible GPU — half your LLM ends up on a
card giq is not scheduling against.

**Not implemented yet:** two *batch* jobs running at once on different cards.
Each card holds its own on-demand instance, but job dispatch is still
serialized, so a render on one card and a batch on the other take turns.
Resident instances (the pinned set) serve concurrently throughout.

## Systemd service

For a server — a dedicated user, hardening, `GIQ_HOME` — use
`deploy/giq.service` and [deployment.md](deployment.md). On a desktop or
development checkout, a user unit is enough:

```bash
# Install service
make install-service

# Control
systemctl --user start giq
systemctl --user status giq
systemctl --user stop giq

# View logs
journalctl --user -u giq -f
```
