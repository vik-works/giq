<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# ADR-003: The domain — engines, weights, recipes, instances

**Status:** Accepted — implemented in full (the migration below)
**Date:** 2026-10-01
**Authors:** giq maintainers
**Amends:** [ADR-002](ADR-002-model-instances.md) (its terms; its decisions on
files, schemas, VRAM and the adapter contract stand)

## Context

ADR-002 introduced weights, engines and instances, and made the instance
file the unit clients call. In use, the words did not hold:

1. **"Instance" names the blueprint, not the running thing.** The YAML file
   is called an instance, and so, informally, is the process it starts. The
   dashboard, the docs and the code all say "instance" for both, and
   nothing at all names a running model: the runner holds `_Resident` and
   `_Slot` objects, the API says "loaded", the dashboard "models in RAM".
2. **Weights are not a thing of their own.** Every file carries its own
   `weights:` block. One checkpoint serving two recipes is described twice
   (`qwen3.8-27b-nvfp4` and `-chat`; `large-v3` and `whisper-large-v3`),
   the storage catalog reports it twice with a "shared with" note, and
   download and delete — acts on files — are offered per recipe.
3. **"Worker" means two things.** The kind of job (`llm`, `text2image`,
   `ocr` — the dashboard already calls it *modality*) and the Python class
   that wraps an engine process (`LLMWorker`, `VLLMWorker`). About 1,300
   occurrences across code, tests, frontend and docs, and readers have to
   guess which is meant each time.
4. **One recipe, two names.** `flux_klein` is two files — `text2image` and
   `image_edit` — over the same weights and the same sd-server. Names are
   unique only per modality, so every key is a `(worker, model)` pair, and
   switching between the two costs a reload of one process into another.
5. **"Model" is overloaded.** It is what a client sends (`model` in the
   OpenAI contract), the registry entry (`ModelSpec`), the files on disk
   ("model weights"), and the running server ("loaded model").

giq has one outside contributor and no installed base to migrate. Breaking
names is cheap now and expensive later.

## Decision

### D1: Six terms, one meaning each

| Term | Is | Is not |
|---|---|---|
| **Engine** | A runtime build that serves weights: llama.cpp, vllm, sd.cpp, a transformers interpreter. Version, path, the formats it accepts, a parameter schema. | — |
| **Weights** | One checkpoint: a file or directory on disk, or a Hugging Face repository at a revision. Format, source, revision, licence, size. | Something a client names. |
| **Recipe** | Weights + engine + params + measured VRAM + the modalities it serves. Its **name** is what a client sends as `model`. | A process. |
| **Instance** | A recipe running on a card: process, port, VRAM, state. | A file. |
| **Residency** | A recipe's policy: **resident** (kept loaded), **on-demand** (loaded for a job, unloaded when idle), or **off** — plus the card it runs on. | A kind of instance. A resident is an instance kept loaded. |
| **Modality** | The kind of job: `llm`, `text2image`, `image_edit`, `ocr`, `stt`, `tts`, `audio`, `embed`, `depth`, `multiview`. Picks the endpoint and the task shape. | The engine adapter, which is an **adapter**. |

"Model" survives in exactly one place: the OpenAI contract, where `model`
is the recipe name and `/v1/models` lists recipes. To a client, a recipe is
a model; inside giq, the word is not used for anything else.

### D2: Recipes

- One YAML file per recipe, named `<name>.yaml`, in the package
  (`giq/recipes/`) and in the operator's directory (`GIQ_RECIPES_DIR`, by
  default `~/.config/giq/recipes`). An operator file with a built-in's name
  replaces it, as today.
- **Names are unique across modalities.** Every key that is
  `(worker, model)` today becomes the recipe name.
- A recipe declares `modalities: [...]` — usually one. `flux_klein` becomes
  one recipe serving `[text2image, image_edit]` on one sd-server: the two
  files launch it identically today, and an edit is a `txt2img` call with a
  reference image. What does differ per modality stays per modality —
  `max_batch` (8 renders, 4 edits) takes either a number or a map keyed by
  modality.
- `capabilities` stays for what a modality's recipe can do within it
  (`chat`, `vision`, tools) — a property of the recipe, not a modality.
- Aliases stay: a recipe may answer to other names, and an alias that
  collides with another recipe's name is a load error.

### D3: Weights are first-class

- The loader builds a **weights inventory** from the recipes: every recipe's
  `weights:` block resolves to a location (an absolute path, or
  `hf:org/repo@revision`), and recipes that resolve to the same location
  share one Weights item. Metadata that disagrees between them (licence,
  format, revision) is a load error rather than whichever file won.
- The inline `weights:` block stays the way a recipe is written — one file
  per new model, as now. Separate weights files are deferred until an
  operator needs weights that no recipe uses yet (a download ahead of the
  recipe, say).
- **Disk acts belong to weights.** Size, presence, delete and (later)
  download are reported and done per Weights item, with the recipes that use
  it listed. Deleting weights a recipe uses leaves that recipe *uninstalled*,
  not removed.

### D4: Instances are first-class

- One type, `Instance`, where `_Resident` and `_Slot` were two. An instance
  knows its recipe, card, port, adapter, state (`starting`, `ready`,
  `stopped`, read from the adapter) and the residency it was started under.
- The runner indexes instances the two ways it schedules them: residents by
  recipe, and the on-demand instance by card, loaded one at a time in queue
  order. Either has a lane of concurrent jobs, as wide as its recipe says.
  (Amended while implementing: the proposal
  was one map, but the two indexes are two scheduling policies, and folding
  them into one changes dispatch for no gain. One type is what makes an
  instance a thing of its own.)
- An instance is named `recipe@card`. Today's rule — at most one instance
  per recipe, and one on-demand instance per card — is a scheduling rule,
  not a property of the type. Running a recipe on two cards is a later
  decision that needs no new concept.
- `GET /instances` lists them. The dashboard's "running" view is this list.

### D5: Residency

- Policy and card binding are per recipe (`recipe_policy`, migrated from
  `model_policy`). `config.yaml` `residents:` names recipes
  (`gemma-4-12b`); the `llm/gemma-4-12b` form is read with a deprecation
  warning.

### D6: Modality and adapters in code

- `WorkerType` becomes `Modality`. The engine wrapper classes become
  adapters (`LlamaCppAdapter`, `VllmAdapter`, `SdCppAdapter`, and the
  child-process adapters); `giq/workers/` becomes `giq/adapters/`.
- `JobRequest.worker` becomes `JobRequest.modality`. `worker` is accepted
  as an input alias, with a deprecation warning, for one minor release.

### D7: The HTTP API

| Today | After |
|---|---|
| `GET /control/models`, `GET /stats/models` | `GET /recipes` — every recipe with its residency, card, fit and whether its weights are present |
| `POST /control/models/{worker}/{model}` (policy), `…/device` | `PUT /recipes/{name}/residency`, `PUT /recipes/{name}/card` |
| `GET /storage`, `DELETE /storage/models/{worker}/{model}` | `GET /weights`, `DELETE /weights/{id}` — `/storage` keeps the per-mount disk report |
| — | `GET /instances` |
| `GET /engines` | unchanged |
| `POST /run`, job bodies | `modality` instead of `worker` (alias accepted) |
| `/v1/*` | unchanged: `model` is a recipe name |
| stats JSON `worker`, `model` | `modality`, `recipe` |

The stats database migrates its columns in place (`worker` → `modality`,
`model` → `recipe`); history is kept.

### D8: The dashboard

Three places, one per layer:

- **Inventory** — Weights (size, source, licence, used by, delete) and
  Engines (version, path, health).
- **Recipes** — today's model cards, slimmer: engine and weights as links
  into Inventory, residency and card, fit, and ADR-002's editor (duplicate,
  edit params from the schema, measure) when it is built.
- **Running** — instances per card: port, VRAM, state, unload. This stays
  on the Overview, whose GPU cards and lanes are built from `/instances`,
  rather than becoming a view of its own: it is the page an operator
  watches, and a second one would show the same rows.

The German glossary (`frontend/src/locales/GLOSSARY.md`) gets the six terms
before any string uses them.

## Consequences

- One word per thing in code, API, dashboard and docs; the glossary is the
  arbiter for new code.
- A breaking release: job bodies, control and storage endpoints, the stats
  JSON, the instances directory and its environment variable all change.
  The deprecated aliases (`worker`, `GIQ_INSTANCES_DIR`, `llm/…` residents)
  cover clients and configs for one minor release.
- `flux_klein` stops reloading when a client alternates generation and edits.
- Storage stops double-reporting shared checkpoints.

## Migration

Each step leaves the suite green and is its own commit series:

1. **Glossary** — this ADR, the German terms.
2. **Recipes** — `giq.instances` → `giq.recipes`, files renamed to
   `<name>.yaml`, `Instance` → `Recipe`, `ModelSpec` folded into the recipe,
   `GIQ_RECIPES_DIR` (old variable honoured with a warning).
3. **Global names and modalities** — keys become recipe names, `modalities`
   lists, `flux_klein` merged; `Modality` replaces `WorkerType`.
4. **Weights** — the inventory, `GET /weights`, delete per weights.
5. **Instances** — the runner's one map, `GET /instances`; adapters renamed.
6. **API and stats** — new routes, aliases, column migration.
7. **Dashboard** — Inventory, Recipes, Running; strings in `en` and `de`.
8. **Docs** — every page in the new terms; ADR-002 marked amended.

## Alternatives considered

- **"Model" for the files.** Collides with the `model` a client sends and
  with `/v1/models`, which lists recipes. Every sentence would need to say
  which model.
- **"Resident" for a running recipe.** Already means "kept loaded by
  policy"; an on-demand instance is running and not resident.
- **"Blueprint" for the recipe.** Pairs well with "instance", reads worse in
  the dashboard ("pin this blueprint"). "Recipe" is ingredients plus method,
  which is what the file is.
- **Weights files now.** A second file per new model for every operator, to
  serve a case (weights with no recipe) nobody has yet. Deferred, not
  refused: the inventory is already keyed by location, so the files can come
  without changing it.
- **Keep `(worker, name)` keys.** Keeps `flux_klein` as two recipes over one
  process and every key a pair. Unique names cost one rename today.

## Open questions

- Download: does `GET /weights` offer fetching `hf:` weights that are
  missing, and with what guard (disk space, licence acceptance)?
- Per-recipe access (which app may call which recipe) — with the access ADR,
  when there is one.
