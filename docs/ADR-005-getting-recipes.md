<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# ADR-005: Getting recipes — show what runs here, plan and fetch the rest

**Status:** Accepted
**Date:** 2026-10-03
**Authors:** giq maintainers

## Context

The catalog lists every recipe giq knows: those whose weights are on this
machine, and those whose weights are not. On a fresh install that is nearly
all of them. The dashboard shows the absent ones as a line of pills under
the installed ones, and `/recipes` returns them with `installed: false`.
Users find this noise: they see what giq could do somewhere, not what it
does here, and nothing on the page helps them close the gap.

Closing the gap is manual today:

- **giq fetches nothing.** A recipe records where its weights came from
  (`weights.source`, `revision`, `licence`) as provenance only. The operator
  downloads by hand and puts the files where `weights.path` says.
- **Half the built-in recipes cannot say where to get their weights.** The
  llama.cpp recipes name a GGUF path but no source; the sd.cpp recipes name
  ComfyUI-style split files with no source; three recipes point into an LM
  Studio directory under `~`. A source can only name a whole repository, so
  one GGUF of a repository that holds twenty cannot be said at all.
- **A vision recipe's projector is not weights.** `params.mmproj` names a
  file the weights inventory, the storage report and deletion never see.
- **Nobody checks whether a recipe can run here before trying.** A recipe
  that needs 26 GB on a 16 GB card, NVFP4 weights on a card without the
  kernels for them, a gated repository without a token, or an engine plugin
  that is not installed fails when it is first loaded, or at download time.
- **What a plugin would add is invisible.** ADR-004 moves image generation,
  speech, OCR and vllm into plugins. Without one installed, its modalities
  and recipes do not exist as far as giq knows. The user does not learn that
  a plugin would add them.

## Decision

### D1: The catalog says, per recipe, whether it runs here

Every recipe gets an **availability**, computed by giq from what it knows
about this machine:

| Availability | Meaning |
|---|---|
| `ready` | Engine present, weights on disk, and it fits a card. |
| `fetchable` | Like `ready`, except the weights are not on disk, and the recipe says where to get them. |
| `manual` | The weights are not on disk, and the recipe does not say where to get them. The operator places the files. |
| `unfit` | It cannot run on this machine as it is: too large for every card, a compute capability no card has, or an engine binary that is missing. |

`/recipes` returns the availability and the reasons behind it, for every
recipe. `/v1/models` and the dashboard's main recipe view show `ready`
recipes only. The rest live in a separate **Add** view (D6) that the user
opens on purpose.

### D2: A recipe's weights say where to fetch them

- `weights.source` (and each part's `source`) is the place to fetch from,
  not only provenance. Hugging Face is the one kind for now. A `url:` kind
  can follow without changing the rest.
  - `hf:org/repo` is the whole repository: a checkpoint directory.
  - `hf:org/repo/path/in/repo` is one file of it: one quantisation of a GGUF
    repository that holds twenty, or one component of a ComfyUI-style repo
    (`hf:Comfy-Org/z_image_turbo/split_files/vae/ae.safetensors`). For the
    first shard of a sharded GGUF, the other shards come with it.
- `weights.revision` pins the commit. A fetch without one takes the current
  head, and the plan (D3) says it is unpinned.
- Where the files go is where giq already looks:
  - With a `path`, under the models directory at that path: a repository's
    files in that directory, one file as that file.
  - Without a `path` (whole repositories only), the Hugging Face cache, as
    the children download them today.
- **The projector becomes a part:** `weights.parts.mmproj`. `params.mmproj`
  stays accepted for one release, with a warning, and is read as that part.
  The projector is then inventoried, sized, fetched and deleted like any
  other weights.
- Every built-in recipe gets a `source` and a `revision`, pinned to the files
  its VRAM figure was measured with. Recipes that predate the models giq is
  tuned for (gemma-3-27b-it-qat, llama-3.2-3b, qwen-coder-30b,
  qwen3.5-35b-a3b, qwen3.6-27b) leave the built-in catalog; an operator who
  still runs one keeps its file in their recipes directory.

### D3: Before anything is fetched, giq plans

`plan(recipe)` returns checks, each `ok`, `warn` or `fail`, with a message a
user can act on:

| Check | From | Fails or warns when |
|---|---|---|
| Plugin | the registry and the plugin index (D5) | the engine or modality is not registered: "needs the giq-sdcpp plugin" with its install command |
| Engine | the engine table | the engine binary is not found: where to get it |
| Card | `vram.gb` against every card's total memory | no card is large enough (fail). It fits only by evicting the resident set (warn). |
| Compute capability | a new optional `Engine.check(recipe, card)` hook | no card has the compute capability the engine needs for these weights, e.g. vllm and NVFP4 (fail) |
| Disk | the files' sizes from the Hub against free space where they go | there is not enough free space (fail) |
| Already here | the weights inventory | (ok) "4.1 of 12.3 GB already on disk": a part another recipe shares |
| Access | the Hub's `gated` flag and whether a token is configured | the repository is gated and there is no token, or the token lacks access (fail) |
| Licence | `weights.licence` and the Hub's licence tag | the licence is non-commercial or custom (warn, shown before the fetch) |
| Pinned | `weights.revision` | there is none (warn) |

Checks that need the Hub degrade without network. Size and access then
read "unknown", and the plan says so; it does not fail.

The plan is one function, used by the CLI, the API and the dashboard:
`giq add <recipe> --dry-run`, `GET /recipes/{name}/plan`, and the Add view.

### D4: giq fetches, by the operator's command or, where allowed, the service's

- **`giq add <recipe>`** prints the plan, asks once, then fetches. It runs as
  the operator, who can write the models directory. It is how a hardened
  install (the systemd unit keeps models read-only) gets weights.
- **`POST /recipes/{name}/fetch`** fetches from the service. It is on when the
  models directory is writable by the service; otherwise it returns 409, and
  the response carries the `giq add` command to run instead. The access
  token guards it, like every other change (`DELETE /weights`).
- **The downloader is not the GPU queue.** Fetches use no GPU, so they don't
  take a lane; they run in a small download service:
  - one fetch at a time per checkpoint;
  - resumable (the Hub library resumes partial files);
  - cancellable;
  - visible at `GET /downloads` with bytes done and bytes total.
  - Files land under a temporary name and appear whole. `installed()` turns
    true only when every file and part is there.
- **Removing a recipe deletes its weights unless another recipe uses them.**
  This builds on `DELETE /weights/{id}`, which already counts users.
- **The Hub token** comes from `HF_TOKEN` or the Hugging Face token file, as
  the children already read it. giq never stores or shows it.

### D5: A curated plugin index, shipped with giq

`giq/plugins.json` lists the curated plugins (ADR-004 D8). It is data only:
each entry gives the plugin's name, package, one-line summary, the
modalities and engines it adds, the recipes it brings, and what it needs
beyond Python (an engine binary, a CUDA level). It ships with each giq
release.

- `giq plugins` lists installed plugins and the curated ones not installed,
  each with its install command.
- The Add view shows a curated plugin that is not installed as an entry that
  says what it adds, and how to install it.
- **The dashboard does not install plugins.** Installing one changes giq's
  own environment: a plugin pulling a different torch can break the running
  service, and the service has to restart to load the plugin. Plugins are
  installed by the operator, from the command line.

### D6: The dashboard: "On this machine" and "Add"

- **On this machine** (the recipes view today) shows `ready` recipes only.
- **Add** lists everything else, grouped by modality, with a search field:
  - `fetchable` and `manual` recipes, each with its plan and either a
    **Get** button or the `giq add` command;
  - curated plugins that are not installed, under the modalities they add;
  - `unfit` recipes, collapsed, with the reason.
- **Fetches in progress** show on the recipe and in the Add view, with
  progress, and can be cancelled. A recipe becomes `ready` and moves to "On
  this machine" when its fetch completes.

### D7: `/capabilities` is the per-modality answer for clients

Clients ask "what can I run for OCR?", not "which recipes exist?".
`/capabilities` already answers per modality, but it lists every recipe
that serves it, ready or not, so a client that picks one from it can get a
recipe with no weights on this machine. It becomes the advertisement:

- `recipes`: the `ready` recipes only, in the order a client should prefer
  them: kept warm first, in the resident set's order, then the on-demand
  ones by name. This is what a client may pass as `model` and expect to
  run. An operator makes a recipe the default by keeping it warm; giq adds
  no ranking of its own.
- `available`: the recipes it could have, each with its availability
  (`fetchable`, `manual`) and the plan's one-line verdict. Clients show it
  ("OCR is not set up here; glm-ocr can be fetched, 4.4 GB") rather than
  failing at the first request. `unfit` recipes are left out.
- `default`: the recipe a request that names none runs on: the first of
  `recipes`. The routes that pick a recipe themselves (`/v1/audio/*`,
  `/llm/endpoint`) use the same choice instead of a fixed name.
- Every registered modality is listed, also with no `ready` recipe, so a
  client learns that the plugin is installed but nothing is fetched yet.
- `GET /capabilities/{modality}` returns one entry; `/capabilities` still
  returns them all.

No new endpoint: `/capabilities` stays the one place a client asks, and
the dashboard reads the same data.

## Consequences

- A fresh install shows what it runs, and the path to everything else is
  one view away, with the reason a recipe can or cannot run here.
- giq starts writing to the models directory: from the CLI always, and from
  the service where the operator allows it. A full disk is checked before
  the fetch, not found halfway.
- Recipes become installable things, so their `source` and `revision` must
  be right. A test checks that every built-in recipe has a
  source and a revision; CI does not fetch.
- `huggingface_hub` becomes a declared dependency of core. Today it only
  arrives with transformers, which leaves core with torch (ADR-004 step 5).
- `params.mmproj` is deprecated in favour of `weights.parts.mmproj`. Operator
  recipes keep working for a release, with a warning.

## Out of scope

- **A remote recipe index** or community recipes. A recipe names arbitrary
  weights and parameters, so trusting a third party's recipe is the same
  question as trusting a third party's plugin (ADR-004, open question).
  The recipes giq ships and the ones plugins bring are the catalog.
- **Installing plugins or engine binaries** from giq (D5).
- **Sources other than the Hugging Face Hub.**

## Migration

1. Schema: file sources (`hf:org/repo/file`), `weights.parts.mmproj`,
   `params.mmproj` read as the part; sources and revisions on every built-in
   recipe; the old recipes go.
2. `plan()` and `GET /recipes/{name}/plan`; availability on `/recipes`;
   `Engine.check` for vllm; `/capabilities` per D7, and the routes that
   pick a recipe use its `default`.
3. The download service, `POST /recipes/{name}/fetch`, `GET /downloads`,
   `giq add`.
4. `giq/plugins.json` and `giq plugins`.
5. Dashboard: "On this machine" and "Add", with fetch progress.
6. Docs: getting recipes, the token, the writable models directory.
