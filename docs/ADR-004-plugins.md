<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# ADR-004: Plugins — a small core, and everything else registered into it

**Status:** Accepted
**Date:** 2026-10-03
**Authors:** giq maintainers

## Context

giq's core job is model-independent: a queue, a scheduler that places
instances on cards and evicts them, a VRAM gate, residency, recipes and
weights, stats, access rules, a dashboard and an OpenAI-compatible API. Yet
every engine and modality it serves is wired into that core by hand, in about
a dozen places:

- **Closed sets.**
  - The `Modality` enum fixes the modalities, and the recipe schema accepts
    no others.
  - `ENGINE_OF_BACKEND` fixes the engines.
  - `MODALITY_ENGINES`, `MODALITY_PARTS` and `ENGINE_PARAMS` fix which engine
    serves what and which parameters it takes.
- **A factory chain.** The runner builds adapters in an `if modality == …`
  chain. It also checks adapter classes by name in several places
  (`SttAdapter`, `VllmParams`), and the start budget has a branch per engine.
- **Named routes.** `/ocr`, `/depth` and `/v1/audio/*` live in core and send
  fixed recipe names (`kokoro`, `whisper-large-v3`, `ecapa-tdnn`), ignoring
  what the client asked for. `/test/{kind}` and `/llm/endpoint` hardcode
  recipes too.
- **Startup and the CLI.** Lifecycle code kills stale `llama-server`,
  `sd-server` and `vllm serve` processes by name. `giq prepare` knows only
  vllm. The port block starts at llama.cpp's historical port.
- **The dashboard** repeats the modality list in six places: types, colours,
  icons, labels, sandbox tabs, engine notes.
- **Dependencies.** Every install pays for every engine: torch, transformers,
  pyannote, speechbrain, kokoro and spacy are hard dependencies, even for a
  server that only runs llama.cpp.

Adding a modality therefore means editing core. Multiview is the case in
point: one project needed it, so every install carried a research dependency
tree and a second interpreter until 0.6.0 removed it. It still needs a way
back that does not go through core.

## Decision

### D1: A plugin is an installed Python package

A plugin is a Python distribution that declares an entry point in the group
`giq.plugins`, naming a `giq.plugin.Plugin` object. Installing the package
into giq's environment is all it takes: giq reads the entry points at
startup. There is no config switch to enable a plugin, and no plugin
directory to scan.

### D2: What a plugin registers

A `Plugin` declares, all optional except its name and API version:

- **Engines.** For each engine:
  - its name;
  - how giq finds it: a default binary, or `self` for giq's own
    interpreter;
  - the environment variable that overrides that location;
  - a version probe;
  - its stale-process pattern, for the startup sweep;
  - its port base, for a server engine;
  - its default start budget;
  - its recipe `params` schema and named profiles;
  - which `request_defaults` keys a recipe may set;
  - an optional extra recipe-validation hook.
- **Modalities.** For each modality:
  - its name and display label;
  - an icon from the dashboard's fixed set (unknown names fall back to a
    generic one);
  - its default lane width and job timeout;
  - the `weights.parts` it reads;
  - the task keys that carry payloads, for the privacy byte count;
  - any `GIQ_*` root override for relative weight paths.
- **Adapter factories:** `(engine, modality) → callable(recipe, device) →
  Adapter`. The factory replaces the runner's `if` chain.
- **Recipes:** a directory of built-in recipe files that ship with the plugin.
- **Routes:** FastAPI routers mounted at startup. Today's `/ocr`, `/depth`
  and `/v1/audio/*` move into the plugins that serve those modalities, and
  take the recipe from the request instead of hardcoding it.
- **CLI:** optional `giq` subcommands (`giq prepare vllm` moves into the vllm
  plugin).
- **A smoke test** per modality, which `/test/{modality}` runs instead of its
  hardcoded table.
- **Dashboard UI** (D6): prebuilt sandbox panels and their strings, served by
  core and loaded into the dashboard at runtime.

### D3: What core keeps

- The queue, `Orchestrator.submit_job` (still the single chokepoint), the
  runner and its residency, eviction and VRAM gate.
- Recipes, weights, the inventory and storage.
- Stats, the policy store, access rules, `/status`, `/gpus`, `/instances`,
  `/recipes`, `/weights` and `/engines`.
- The dashboard shell.
- **The `llm` modality and the OpenAI chat API** (`/v1/chat/completions`,
  `/v1/responses`, `/v1/models`). Chat is what most clients come for, and its
  routes span every LLM engine, so core defines `llm`, the `ServedLLM`
  contract and the chat routes.
- **The llama.cpp engine.** llama-server is a binary, so it costs giq no
  Python dependencies, and with it a plain `giq` install serves GGUF chat out
  of the box. It registers through the same registry as everything else,
  from inside core. vllm, the heavy LLM engine, is a plugin.
- **The `transformers` engine name.** OCR and depth both run on it, and two
  plugins registering one engine would clash (D5). Core declares the name
  only, as "runs in giq's own interpreter". The library itself is a
  dependency of each plugin that uses it, so core still needs no torch.

Core defines the adapter contracts:
- the `Adapter` protocol the runner already relies on;
- `ServedLLM` for engines that serve chat over HTTP;
- `SubprocessAdapter` for models in a child process.

The runner stops checking adapter classes by name: whatever it needs to know
comes from the adapter or from its registered engine (whether it runs inside
giq's process, its lanes, its start budget).

### D4: Child processes in their own interpreter

A plugin whose dependencies clash with giq's runs its model in a child on a
declared interpreter, the way vllm does today. The child side of the
protocol is a small package that imports only the standard library:
`giq_child` (ready/results/error lines, the `run_batch` loop). It is a
top-level package of its own rather than part of `giq`, so a foreign
interpreter can have it without any of giq. It ships in giq's wheel and is
published on its own too, as `giq-child`. A foreign interpreter
installs `giq-child` and the plugin's own package, never giq's source tree
and never a `PYTHONPATH` override. That removes for good the problem that
blocked wheel installs.

### D5: Versioned contract, and plugins only add

`giq.plugin.API_VERSION` is an integer. A plugin states the version it was
written for. A plugin built for another version is refused at startup with a
clear log line, and `/status` lists it under `plugins` with the reason. One
broken plugin does not stop giq or the other plugins. Every plugin that loads
is also listed, with its version and what it registered.

Plugins add; they never override. A plugin that registers a route, engine,
modality or recipe name that core or an earlier plugin already holds is
refused at startup, with the clash named.

### D6: The dashboard renders what the registry says, and plugins bring panels

`/capabilities` carries each modality's label and icon name. The dashboard
takes colours, icons and labels from there:
- the eight series colours go to modalities in registration order, and the
  rest share the neutral colour;
- a modality the dashboard does not know gets a generic icon and the label
  the server sends.

**Sandbox panels are plugin UI.** The dashboard shell keeps only the panels
for core's own modality, `llm`: chat, tools and vision. Every other panel
ships with the plugin that serves its modality:
- text to image and image edit in `giq-sdcpp`;
- speech recognition, text to speech and voiceprint in `giq-speech`;
- whatever a third-party plugin brings.

The contract:
- **What a plugin ships.** A plugin's UI is a prebuilt ES module plus an
  optional stylesheet and its strings (`en`, `de`, any other language). They
  are declared in a `ui/manifest.json` inside the plugin package. Core serves
  them at `/plugins/<name>/ui/` and lists them in `/capabilities`.
- **How the dashboard loads it.** It `import()`s each plugin's module when the
  sandbox opens. The module exports panels, each with an id, the modality it
  exercises, an icon name and a React component. The tabs, the "Test in
  sandbox" links from the Recipes view, and the disabled-with-a-reason
  states all follow the registry, so they are no longer a fixed table.
- **What the host provides.** React, the API client, i18next, the format
  helpers and the shared components come in through a small versioned host
  object. A plugin never bundles its own React; two copies would break
  hooks. A `@giq/plugin-ui` package carries the types and a Vite preset that
  marks those modules as external.
- **Rules.** A plugin's CSS uses the dashboard's tokens and prefixes its
  classes with `pl-<plugin>-`, the same namespacing rule views follow. Its
  strings load into the i18next namespace `plugin-<name>`, and a missing
  language falls back to English. The UI contract is versioned with
  `API_VERSION`.
- **Trust.** A plugin's UI runs same-origin with the dashboard. It is code the
  operator installed, trusted exactly as its Python is. Server text inside a
  panel is still rendered as text, never as HTML.

### D7: Packages and repositories

Core and the curated plugins live in this repository as one uv workspace and
are released together:

| Package | Registers | Brings |
|---|---|---|
| `giq` | core, `llm` modality, chat API, engines `llama.cpp` and `transformers` (the name), chat/tools/vision panels | fastapi, pydantic, httpx, pyyaml — no torch |
| `giq-vllm` | engine `vllm`, `giq prepare vllm` | nothing in giq's env (vllm runs in `envs/vllm`) |
| `giq-sdcpp` | engine `sd.cpp`, modalities `text2image`, `image_edit`, their panels | nothing (sd-server is a binary) |
| `giq-speech` | engines `faster-whisper`, `faster-whisper+pyannote`, `speechbrain`, `kokoro`; modalities `stt`, `audio`, `embed`, `tts`; `/v1/audio/*`; their panels | torch, faster-whisper, pyannote, speechbrain, kokoro |
| `giq-ocr` | engine `transformers` (OCR children), modality `ocr`, `/ocr` | torch, transformers, pypdfium2, opencv |
| `giq-depth` | modality `depth`, `/depth` | torch, transformers |
| `giq-defaults` | nothing itself | depends on all of the above |

Each plugin brings its own built-in recipes; core brings the llama.cpp ones.
`giq-defaults` is what the documented install and `install-debian.sh`
install, so a normal install serves what it serves today. A server that only
needs llama.cpp installs `giq` alone, with no torch. Plugins are
packages rather than extras, because `uv sync` prunes extras it was not told
about; that rule from before 0.6.0 still holds.

Plugins outside the curated set live in their own repositories:
`giq-multiview` first, as the worked example, and whatever third parties
write.

### D8: The curated set has a bar

A plugin joins the curated set when:
- the modality is broadly useful;
- its licences allow commercial use;
- it runs on giq's shared torch line, or in its own server process;
- it has a measured VRAM figure.

Everything else is a plugin in its own repository. The curated set is a
promise of quality, not a list of everything that works.

## Consequences

- **Core shrinks** to the scheduler and its API, and a minimal install needs
  no torch.
- **Adding a modality never touches core.** Multiview comes back as
  `giq-multiview`, maintained where it is needed.
- **The built-ins prove the contract.** They register through the same
  interface third parties use, so it cannot rot unnoticed.
- **A plain install is useful on its own:** `giq` alone serves GGUF chat
  through llama.cpp.
- **The dashboard becomes extensible:** a plugin brings its own sandbox
  panels, so a new modality is testable in the browser without a dashboard
  release.
- **One release, more packages.** The workspace releases core and the
  curated plugins together; CI builds and tests all of them.
- **Behaviour changes:**
  - The audio routes stop ignoring `model`: `/v1/audio/transcriptions`
    honours the recipe the client names, with today's recipe as the default.
  - `/test/{modality}` covers whatever is installed.
  - `/status` gains `plugins`.
- **Breaking, for code that imports giq:** adapters move out of
  `giq.adapters` into their plugins' packages.

## Migration

Each step leaves the suite green:

1. **Registry in core.** Built-in engines and modalities register through
   it, still inside the current package. The enum, the factory chain, the
   schema tables and the class checks give way to registry lookups. Nothing
   changes for users.
2. **Child protocol.** `giq_child` becomes standalone and stdlib-only;
   children import only it.
3. **Routes, CLI and smoke tests** move behind registrations, and the
   hardcoded recipe names go.
4. **Dashboard.** It reads modality metadata from `/capabilities`. The UI
   plugin host and `@giq/plugin-ui` follow. The curated panels move out of
   the shell into their plugins' UI bundles, still in this repository.
5. **Workspace.** The curated plugins split into packages under `plugins/`,
   with `giq-defaults`. Torch leaves core's dependencies.
6. **`giq-multiview`** in its own repository, against the published
   contract.
7. **Docs.** "Writing a plugin", the curated list, and the install paths
   (`giq-defaults`, or core plus a chosen few).

## Alternatives considered

- **Plugins as extras of one package.** `uv sync` prunes extras it was not
  told about, which once silently removed an installed stack while
  `/capabilities` kept advertising it. Separate packages cannot be pruned
  that way.
- **One repository per curated plugin.** Every contract change would mean a
  lockstep release across seven repositories, for no gain while one team
  maintains them all.
- **Defaults built in, plugins only for the rest.** That means two code
  paths, and the plugin path would rot unseen.
- **Panels for curated modalities only, nothing for third parties.** That
  is simpler, but a third-party modality could not be tried in the browser,
  and the curated panels would stay a fixed table in the shell, the kind of
  hardwiring this ADR removes.
- **Module federation for UI plugins.** It solves sharing React between
  separately built bundles, but brings a build-time coupling and a runtime
  of its own. A host object plus externals in the plugin's build is enough
  for panels.

## Open questions

- Signing or pinning of third-party plugins, for an install that must not
  load arbitrary code. Today anything installed into giq's environment is
  trusted, as any Python dependency is.
