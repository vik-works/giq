# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""What a giq plugin declares (ADR-004).

A plugin is a Python package with an entry point in the group
``giq.plugins`` that names a :class:`Plugin`. giq reads every such entry
point at startup; installing the package is the whole of enabling it. Core
registers its own engine and modality the same way, so the built-ins and a
third party's plugin go through one contract.

This module is the contract and nothing else: plain declarations, importable
without importing any engine. :data:`API_VERSION` changes when a field's
meaning does, and a plugin written for another version is refused at
startup rather than half-loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_VERSION = 1

# The runtime of an engine whose adapter runs inside giq's own interpreter
# (in-process, or as a child on sys.executable) rather than as a binary.
SELF = "python"


@dataclass(frozen=True)
class Binary:
    """Where an engine's executable is, and how to ask it what it is."""

    # The executable when nothing declares one: a bare name looked up on
    # PATH, or a path. A callable is resolved when the table is built, not
    # at import, so env and PATH set after import still count.
    default: str | Callable[[], str]
    # Environment variable that outranks config.yaml's `engines:` entry.
    env: str | None = None
    version_args: tuple[str, ...] = ("--version",)
    detail: str = ""


@dataclass(frozen=True)
class AdapterContext:
    """What an adapter factory is told: which recipe, on which card."""

    recipe: str
    device: str | None
    # A one-off weights path for this start (llama.cpp only, from /run).
    model_path: str | None = None


# (context) -> an adapter the runner can start, run and stop.
AdapterFactory = Callable[[AdapterContext], Any]


@dataclass(frozen=True)
class Engine:
    """A runtime recipes can name in `engine:`.

    Everything here is about the engine, not one model: how it is found,
    what parameters a recipe may give it, how long a start may take. What a
    recipe's model needs comes from the recipe.
    """

    name: str
    # None: runs in giq's own interpreter (SELF).
    binary: Binary | None = None
    detail: str = ""
    # Old spellings of the name, accepted with a warning.
    aliases: tuple[str, ...] = ()
    # The schema a recipe's `params` is validated against: a subclass of
    # giq.recipes.schema.EngineParams. None takes no parameters.
    params: type[Any] | None = None
    # Named parameter sets a recipe can pick with `profile:`.
    profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # Keys a recipe's `request_defaults` may set: None for any, an empty set
    # for none.
    request_defaults: frozenset[str] | None = frozenset()
    # Extra checks on a whole recipe using this engine; raises ValueError.
    validate: Callable[[Any], None] | None = None
    # A recipe's VRAM figure may be given as weights + KV + overhead, which
    # this derives into vram.gb from (vram dict, validated params).
    derive_vram: Callable[[Mapping[str, Any], Any], float | None] | None = None
    # Concurrent requests the engine itself runs for these params, which
    # then decides the recipe's lane width; None leaves it to the modality.
    lanes: Callable[[Any], int | None] | None = None
    # Seconds a start of this recipe may take; None is the runner's default.
    start_budget: Callable[[Any], float | None] | None = None
    # A process of this engine left over from a run that died, as a regex for
    # `pkill -f` with `{port}` for each of giq's server ports; swept at
    # startup. Qualified by port so it only ever reaches giq's own servers.
    stale_pattern: str | None = None
    # Further cleanup at startup, given giq's server ports (vllm stops the
    # systemd scopes its servers ran in).
    sweep: Callable[[set[int]], None] | None = None
    # `giq prepare <engine> …`: one-off build steps before it serves; takes
    # the remaining arguments, returns an exit code.
    prepare: Callable[[list[str]], int] | None = None
    # Why a card cannot run this recipe, given the card's compute capability
    # ("12.0"), or None when it can. A recipe no card can run is shown as
    # unfit for this machine (ADR-005).
    check: Callable[[Any, str], str | None] | None = None
    # VRAM to keep free on top of a recipe's figure when it loads on demand,
    # in GB; None is the scaled default (up to 2 GB, against the compute
    # buffers an engine grows into). An engine that reserves everything at
    # start and never grows past its declared figure needs only a little.
    vram_margin: float | None = None
    # The context window an LLM recipe is served with, from its validated
    # params; None leaves it to llama.cpp's tables.
    context: Callable[[Any], int | None] | None = None
    # How an LLM recipe on this engine thinks: "on", "off" or "template"
    # (the chat template decides). None leaves it to llama.cpp's tables.
    reasoning: Callable[[Any], str | None] | None = None

    @property
    def runtime(self) -> str:
        """The executable that runs it, by engine name, or SELF."""
        return self.name if self.binary is not None else SELF


@dataclass(frozen=True)
class SmokeTest:
    """A canned end-to-end job for one modality, for POST /test/{modality}."""

    tasks: list[dict[str, Any]]
    # Seconds to wait for it, a cold start included.
    timeout: float
    # What of the first result to show: result dict -> summary dict.
    summary: Callable[[dict[str, Any]], dict[str, Any]] = lambda result: {}
    # Loading it evicts the resident set for minutes; the caller must confirm.
    evicts: bool = False


@dataclass(frozen=True)
class Modality:
    """A kind of job (ADR-003): llm, ocr, tts …"""

    name: str
    # What the dashboard calls it; core's own strings outrank this for the
    # modalities it knows.
    label: str = ""
    # One of the dashboard's icon names; an unknown one gets a generic icon.
    icon: str = ""
    # Concurrent jobs on a resident's lane when the recipe says nothing.
    lane_width: int = 1
    # Seconds a job may run; None is the runner's default.
    job_timeout: float | None = None
    # The `weights.parts` its adapters read; any other part is refused.
    parts: frozenset[str] = frozenset()
    # Task keys that carry a payload (an image, audio, a document), counted
    # as blobs by the privacy-preserving request log, never read.
    payload_keys: tuple[str, ...] = ()
    # Environment variable that replaces the models directory as the root
    # of this modality's relative weight paths.
    weights_root_env: str | None = None
    smoke_test: Callable[[], SmokeTest] | None = None


@dataclass(frozen=True)
class Plugin:
    """One plugin's registrations."""

    name: str
    # The API_VERSION this plugin was written for.
    api_version: int
    version: str = ""
    engines: tuple[Engine, ...] = ()
    modalities: tuple[Modality, ...] = ()
    # (engine, modality) -> factory. Which pairs exist is also which engines
    # can serve which modality.
    adapters: Mapping[tuple[str, str], AdapterFactory] = field(default_factory=dict)
    # A directory of built-in recipe files that ship with the plugin.
    recipes: Path | None = None
    # FastAPI routers to mount, as "module:attribute" import paths, so that
    # building the registry imports no route code. A route (method and path)
    # another plugin or core already serves refuses the plugin.
    routers: tuple[str, ...] = ()
    # The plugin's dashboard UI: a directory holding `manifest.json`, a
    # prebuilt ES module, its stylesheet and strings (ADR-004 D6). Served at
    # /plugins/<name>/ui/ and listed in /capabilities.
    ui: Path | None = None
