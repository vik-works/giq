<!--
SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)

SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

Guidance for AI coding agents (and humans) working on giq. `CLAUDE.md` imports
this file; keep agent instructions here, in one place.

## What giq is

A GPU inference queue: one FastAPI service that owns the GPU(s) and serves
LLM, image, OCR, depth, audio and embedding recipes behind a job
queue and an OpenAI-compatible API. The terms — engine, weights, recipe,
instance, residency, modality — are ADR-003's
(`docs/ADR-003-domain.md`); use them, one meaning each, in code, API,
dashboard and docs. `Orchestrator.submit_job`
(`giq.services.orchestration`) is the single chokepoint — every request path,
including tool-calling chat, goes through it. Keep it that way.

## Layout

- `src/giq/api/` — HTTP routers (`router.py` job API and `/instances`, `recipes_api.py` `/recipes`, `openai_compat.py`, `openai_responses.py`, `stats_api.py` stats, `/storage` and `/weights`, `access.py`)
- `src/giq/services/orchestration.py` — `Orchestrator`, the one way into the queue
- `src/giq/runner.py` — the scheduler: residency, eviction, and the instances it starts and stops
- `src/giq/recipes/` — the built-in recipes, one YAML file each (`<name>.yaml`: modalities, weights, engine, parameters, residency, measured VRAM, and the reasoning as comments), plus the schema and loader; operator files in `GIQ_RECIPES_DIR` add or replace recipes (ADR-002, ADR-003 for the terms)
- `src/giq/registry.py` — the catalog: recipes by name (aliases resolved), and the default resident set
- `src/giq/weights.py` — where a recipe's weights are: its `weights.path`/`weights.parts`, under the models directory, with the env overrides, and the weights inventory; every adapter and the storage report ask here
- `src/giq/adapters/` — the engine adapters, one module per engine or modality (`llama_cpp.py`, `vllm.py`, `sdcpp.py`, `stt.py` …); `_*_child.py` run in subprocesses
- `src/giq/paths.py` — every filesystem location, resolved from env > `config.yaml` `paths:` > `GIQ_HOME` > defaults
- `deploy/` — the systemd unit and Debian install script (see `docs/deployment.md`)
- `frontend/` — the dashboard: React + TypeScript + Vite, built into `src/giq/static/ui/` (gitignored, shipped in the wheel) and served at `/dash`. `src/components/` shared UI and charts, `src/state/` the app-wide pollers, `src/api/` the typed client, `src/views/<view>/` one folder per page, `src/locales/<lang>/<ns>.json` strings
- `envs/vllm/` — the vllm engine's own uv project: vllm pins its torch, transformers and fastapi, so it runs as a separate server process on its own interpreter, like llama-server
- `tests/` — pytest; `docs/` — user docs (API, configuration, access and privacy, engines, development), ADRs, and `docs/images/` (README screenshots)

## Commands

```bash
make sync     # uv sync and the dashboard (envs/vllm: `cd envs/vllm && uv sync`)
make test     # pytest
make check    # ruff + ty
make fmt      # ruff format
uvx reuse lint
make ui       # build the dashboard (frontend/ → src/giq/static/ui/)
make ui-dist  # ...and pack it as dist/giq-ui-<version>.tar.gz for a server
tests/deploy/test-ui-install.sh   # the installer's dashboard step, no root
make ui-dev   # Vite dev server proxying to the giq on :8084
cd frontend && npx tsc --noEmit && npm test
```

## Rules

- **Don't run anything on the GPU unless asked.** No live inference and no
  GPU-marked tests (`tests/test_dirty_workers_integration.py`): the machine is
  usually serving, and loading a model evicts the resident set. The CPU suite
  is `pytest --ignore=tests/test_dirty_workers_integration.py`.
- **Don't start a second full giq instance next to a live one.** Its startup
  kills stale `llama-server`/`sd-server` processes, which are the live
  instance's. To exercise routes, mount the routers on an app without the
  lifespan on a spare port.
- **No hardcoded machine paths.** Filesystem defaults go through `giq.paths`
  and stay overridable by a `GIQ_*` environment variable; document new ones in
  the environment-variable table in `docs/configuration.md`.
- **Every new file gets a REUSE header:**
  `reuse annotate --copyright "vikworks UG (haftungsbeschränkt)" --license Apache-2.0 <file>`.
  `reuse lint` must pass.
- **Comments explain why, in the present tense.** Dense rationale comments are
  the house style. No working-journal residue: no session notes, handoffs,
  "turning this on now", dated diary entries, or names of private projects,
  hosts and people. A measurement is welcome as a fact ("Measured on an RTX
  5090: 8.3 GiB peak"), not as a story. Notes meant for the next agent belong
  in this file, not in code or extra markdown files.
- **VRAM figures are measured, not guessed.** A recipe with
  `vram.measured: true` carries a number observed on real hardware; otherwise
  leave it unmeasured and say it's an estimate.
- **Frontend:** one component per file (aim under 250 lines), each with its
  own co-located CSS; colours only from tokens (`frontend/src/styles/tokens.css`
  over Nocturne), never literals. Every user-facing string goes through
  i18next, added to `en` and `de` in the same change, with German terms from
  `frontend/src/locales/GLOSSARY.md`; plurals via `_one`/`_other` keys, numbers
  and dates via `lib/format.ts` (Intl). Server text is rendered as text, never
  through `dangerouslySetInnerHTML`. API calls go through `src/api/client.ts`;
  shared data comes from the `src/state/` hooks rather than a new poller.
- **Frontend CSS is global, so class names are namespaced.** A class a view
  defines carries its view's prefix — `ov-` (overview), `rc-` (recipes),
  `inv-` (inventory), `us-` (usage), `sbx-` (sandbox) — and a new view picks
  its own. Unprefixed
  classes belong to `src/components/` (named after the component:
  `.model-label`, `.filebox`) and to `src/styles/` (Nocturne's vocabulary plus
  base.css utilities: `.hint`, `.warn`, `.field-error`, `.error-box`,
  `.text-good`/`.text-critical`, `.btn-sm`, `.section-title`, `.grid-*`). A
  view may style a shared class only under its own prefix
  (`.ov-mir-table .tag`), never bare. Something two views need moves to
  `components/` or `lib/` rather than being copied. Views load lazily, one
  chunk each with their `locales/<lang>/<view>.json`; only `common` ships
  with the shell.
- **Commits:** conventional style (`feat(ocr): …`, `fix: …`, `docs: …`), one
  theme per commit, no AI attribution trailers.
