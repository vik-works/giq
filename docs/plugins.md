<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# Plugins

giq is a small core with plugins registered into it
([ADR-004](ADR-004-plugins.md)).

**Core** does the model-independent work:
- the queue, the scheduler and the VRAM gate;
- recipes and weights;
- stats, access rules and the dashboard;
- the OpenAI-compatible chat API;
- LLMs through llama.cpp.

**Plugins** bring everything else: further engines (vllm, sd.cpp, the
speech and OCR libraries), further modalities (images, speech, documents,
depth), their routes, their recipes and their dashboard panels. A plugin is
an installed Python package. Installing it is the whole of enabling it;
giq reads plugins at startup.

```bash
giq plugins              # what is installed, and the curated plugins that are not
curl -s localhost:8084/status | jq .plugins
```

## The curated plugins

They live in this repository, are released with core and carry the same
version.

| Package | Adds | Needs besides Python |
|---|---|---|
| `giq-vllm` | engine `vllm` for LLMs: Hugging Face checkpoints, continuous batching, MTP speculation; `giq prepare vllm` | vllm's own interpreter (`envs/vllm`), see [engines.md](engines.md#vllm) |
| `giq-sdcpp` | engine `sd.cpp`; modalities `text2image`, `image_edit`; their sandbox panels | the `sd-server` binary |
| `giq-speech` | modalities `audio`, `stt`, `tts`, `embed`; `/v1/audio/*`; their sandbox panels | torch; a Hugging Face token for pyannote's diarization |
| `giq-ocr` | modality `ocr`; `POST /ocr` | torch, transformers |
| `giq-depth` | modality `depth`; `POST /depth` | torch, transformers |
| `giq-defaults` | all of the above | |

Each brings its own built-in recipes, so a recipe appears in the Add page
once its plugin is installed (see [Getting a recipe's
weights](configuration.md#getting-a-recipes-weights)).

**Outside the curated set.** `giq-multiview` (multi-view geometry with Depth
Anything 3: N images of one scene in, per-view depth and camera poses out)
lives in its own repository. It was written against this contract alone,
and is the example to start from.

## Installing

- **Everything:** a checkout's `make sync` (plain `uv sync`) installs core,
  every curated plugin and the dev tools. `deploy/install-debian.sh` does
  the same. From release wheels, install `giq` and `giq-defaults` with their
  plugin wheels.
- **Core alone:** install the `giq` wheel by itself. It serves LLMs through
  llama.cpp, has no torch, and needs about 40 MB.
- **Core plus a few:** core's wheel and the plugin wheels you want:

  ```bash
  uv pip install --python /opt/giq/.venv/bin/python giq_sdcpp-0.6.0-py3-none-any.whl
  sudo systemctl restart giq
  ```

The dashboard never installs a plugin. Installing one changes giq's own
interpreter, and a plugin pulling another torch could break the running
service. The Add page shows a missing curated plugin's install command
instead.

**When a plugin is refused.** A plugin is refused whole, never half-loaded,
when:
- it was written for another plugin API version;
- its package does not import;
- it registers a name that is already taken (an engine, an alias, a
  modality, an adapter pair);
- a route of its own is already served.

`giq plugins` and `/status` give the reason, and the other plugins and core
keep serving.

## Writing a plugin

A plugin is a package whose entry point in the group `giq.plugins` names a
`giq.plugin.Plugin`:

```toml
# pyproject.toml
[project]
name = "giq-example"
dependencies = ["giq>=0.6.0,<0.7"]

[project.entry-points."giq.plugins"]
giq-example = "giq_example:plugin"
```

```python
# src/giq_example/__init__.py
from pathlib import Path

from giq.plugin import API_VERSION, AdapterContext, Engine, Modality, Plugin

def _adapter(ctx: AdapterContext):
    from giq_example.adapter import ExampleAdapter  # imported when a recipe starts

    return ExampleAdapter(ctx.recipe, ctx.device)

plugin = Plugin(
    name="giq-example",
    api_version=API_VERSION,
    version="0.1.0",
    engines=(Engine("example-engine"),),
    modalities=(Modality("example", label="Example", icon="vector"),),
    adapters={("example-engine", "example"): _adapter},
    recipes=Path(__file__).parent / "recipes",
    routers=("giq_example.api:router",),
)
```

Keep the module that declares the plugin light. giq imports it at startup,
and every engine and ML library belongs inside the adapter factory.

### What a plugin registers

| Field | What |
|---|---|
| `engines` | `Engine`s recipes can name. `binary` is a `Binary` (default path, the `GIQ_*` variable that overrides it, the arguments that print its version) for an engine that is an executable; leave it out for one that runs in giq's interpreter. `params` is the schema of a recipe's `params` (a `giq.recipes.schema.EngineParams`), with `profiles`, `request_defaults`, `validate`. Hooks: `lanes`, `start_budget`, `derive_vram`, `context`, `reasoning`, `check` (whether a card's compute capability can run it), `stale_pattern` and `sweep` (startup clean-up of its leftover processes), and `prepare` (`giq prepare <engine>`). |
| `modalities` | `Modality`s: the job kinds the plugin adds. Each has a `label` and `icon` for the dashboard, a default `lane_width`, a `job_timeout`, the `parts` its recipes' weights may name, its `payload_keys` (counted, never logged), an optional `weights_root_env`, and a `smoke_test` for `POST /test/{modality}`. |
| `adapters` | `(engine, modality) -> factory`. The factory gets an `AdapterContext` (recipe, card) and returns what the runner starts, runs batches on and stops. Which pairs exist is also which engines serve which modality. |
| `recipes` | A directory of built-in recipe files. A recipe whose name or alias is taken is left out and logged: plugins add recipes, they never replace one. |
| `routers` | FastAPI routers as `"module:attribute"`. A route whose method and path are already served refuses the plugin. |
| `ui` | Its dashboard panels (below). |

`api_version` is the `giq.plugin.API_VERSION` the plugin was written for.
It rises when a field's meaning changes, and a plugin written for another
version is refused rather than half-loaded. New optional fields do not
raise it.

### Child processes

Model code usually runs in a child process: stopping the child gives its
VRAM back. Subclass `giq.adapters._subprocess.SubprocessAdapter` and set
`child_module`. The child speaks giq's protocol through `giq_child`, which
is standard library only. A child can therefore run on an interpreter of its
own, without giq installed, when its dependencies cannot share giq's (a
numpy pin, another torch). Put `giq_child`'s directory and the child's own
package on its `PYTHONPATH`, and have an engine `prepare` hook build that
interpreter. `giq-multiview` does exactly this.

### Dashboard panels

A plugin can bring sandbox panels. `ui` names a directory holding:
- `manifest.json`: the API version, the module, its stylesheets, its
  strings per language, and its panels (id, modality, icon, the label's
  key);
- the module itself, which exports `panels`, one React component per id.

Core serves the directory at `/plugins/<name>/ui/` and lists it in
`/capabilities`. The sandbox draws the tab from the manifest, and imports
the module when the tab first opens. A panel that fails to load shows a
note in its place.

A panel gets the recipes that serve its modality (`options`), the chosen
one (`model`) and `onModel`. The dashboard lends it React, react-i18next,
the API client, the format helpers, shared components and the sandbox's
building blocks through `@giq/plugin-ui`. A plugin never bundles its own
React: two copies break hooks.

`@giq/plugin-ui`'s Vite preset builds the module with those imports pointed
at the dashboard (see `frontend/plugins/giq-sdcpp/` for a whole one). It is
not published yet; build against a giq checkout's `frontend/plugin-ui/`.

Rules, as for the dashboard's own views:
- prefix CSS classes `pl-<plugin>-` and use the dashboard's colour tokens;
- the sandbox's layout classes (`sbx-panel`, `sbx-form`, `sbx-row`,
  `sbx-actions`) are yours to use;
- strings go in the namespace `plugin-<name>`, at least in English, which
  is the fallback;
- render server text as text.

A plugin's UI runs same-origin with the dashboard. It is code the operator
installed, trusted exactly as its Python is.

### Testing

`giq-multiview`'s tests run without a GPU, a model or its interpreter:
- the child is replaced by a script that answers in the protocol;
- the route runs against a stub orchestrator;
- one test parses the child's imports, to prove it needs nothing of giq.

The curated plugins' tests sit in `plugins/<name>/tests/` and run with
core's suite (`make test`).
