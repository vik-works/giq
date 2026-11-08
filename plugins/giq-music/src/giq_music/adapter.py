# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-music and text-to-sound workers: ACE-Step 1.5 and Stable Audio 3,
each in a child process on its own interpreter.

The child (``giq_music_child._music_child``) loads the recipe's DiT, VAE and
text encoder, and the 5Hz planner LM, from the recipe's weights.path, which
is ACE-Step's own checkpoint layout (the folders AceStepParams names), and
a DiT that comes as a repository of its own from weights.parts.dit. The
hub is disabled and the layout checked before loading, so nothing is ever
fetched. Per task the planner (when the
task asks for it) fills in what the request leaves open — tempo, key,
duration, the song's structure — and the DiT renders the track.

Sleepy model: loads on demand, evictable. Runs on the acestep interpreter,
like the vllm engine's own: the child imports only the top-level
``giq_music_child`` package plus ``giq_child``, never giq.

Stable Audio 3 (``giq_music_child._sa3_child``) loads one model folder —
the DiT with its autoencoder, and the T5Gemma text encoder beside it — and
renders a prompt to 44.1 kHz stereo; no planner, no lyrics. It serves both
text2music and text2sfx: which one is the recipe's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from giq.adapters._subprocess import SubprocessAdapter
from giq.registry import get_recipe, vram_for
from giq.weights import path_of, require_path
from giq_music.params import AceStepParams, StableAudioParams

# Upper bound for an unregistered model; the registry has the measured ones.
MUSIC_VRAM_GB = 16.0


class MusicResult(BaseModel):
    """One track, and what the model settled on for it.

    ``bpm``, ``key_scale``, ``time_signature`` and ``duration`` are the
    request's values where it gave them, the planner's otherwise; ``lyrics``
    and ``caption`` likewise (the planner may rewrite a short caption)."""

    id: str
    audio_b64: str | None = None
    format: str | None = None
    sample_rate: int | None = None
    duration: float | None = None
    seed: int | None = None
    bpm: int | None = None
    key_scale: str | None = None
    time_signature: str | None = None
    caption: str | None = None
    lyrics: str | None = None
    error: str | None = None


@dataclass
class MusicConfig:
    model: str = "ace-step-1.5-turbo"


class _AudioChildAdapter(SubprocessAdapter):
    """What both engines' adapters share: the recipe's weights folder, the
    child on the engine's own interpreter, MusicResults back."""

    modality: ClassVar[str] = "text2music"
    # Which declared engine's interpreter runs the child.
    engine: ClassVar[str] = ""

    def __init__(self, config: MusicConfig, device: str | None = None):
        super().__init__(config, device)
        # Unknown model or missing weights: fail here, not at spawn.
        self.weights = require_path(config.model)

    @property
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=MUSIC_VRAM_GB)

    def _command(self) -> list[str]:
        from giq.engines import require_binary

        return [require_binary(self.engine), "-u", "-m", self.child_module, *self.child_args()]

    def _spawn_env(self) -> dict[str, str]:
        env = super()._spawn_env()
        # The engine's interpreter has the engine installed but not giq: the
        # plugin's child package and the giq_child protocol travel on
        # PYTHONPATH, so the child imports no giq.
        import giq_child

        path = os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[1]),
                str(Path(giq_child.__file__).resolve().parent.parent),
            ]
        )
        env["PYTHONPATH"] = path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[MusicResult]:
        raw_results = await super().run_batch(tasks, params)
        return [hydrate(r) for r in raw_results]


class MusicAdapter(_AudioChildAdapter):
    """ACE-Step text to music in a child process on the ``acestep`` interpreter."""

    child_module: ClassVar[str] = "giq_music_child._music_child"
    engine: ClassVar[str] = "acestep"
    # A batch of long songs with the planner on outlasts the default 600 s;
    # just under the modality's 900 s job_timeout, so a stuck child is
    # reported as such before the job is cut off.
    run_batch_timeout: ClassVar[float] = 870.0

    def __init__(self, config: MusicConfig, device: str | None = None):
        super().__init__(config, device)
        # The folders of the checkpoint root to load.
        recipe = get_recipe(config.model)
        self.params = (
            recipe.params
            if recipe and isinstance(recipe.params, AceStepParams)
            else AceStepParams()
        )

    def child_args(self) -> list[str]:
        p = self.params
        args = ["--model", self.config.model, "--weights", self.weights, "--dit", p.dit]
        # A DiT published as a repository of its own is weights.parts.dit.
        if dit_path := path_of(self.config.model, "dit"):
            args += ["--dit-path", dit_path]
        args += ["--steps", str(p.steps), "--guidance", str(p.guidance)]
        args += ["--tasks", ",".join(p.tasks)]
        return args + (["--lm", p.lm] if p.lm else [])


class StableAudioAdapter(_AudioChildAdapter):
    """Stable Audio 3 in a child process on the ``stable-audio`` interpreter."""

    child_module: ClassVar[str] = "giq_music_child._sa3_child"
    engine: ClassVar[str] = "stable-audio"

    def __init__(self, config: MusicConfig, device: str | None = None):
        super().__init__(config, device)
        recipe = get_recipe(config.model)
        self.params = (
            recipe.params
            if recipe and isinstance(recipe.params, StableAudioParams)
            else StableAudioParams()
        )

    def child_args(self) -> list[str]:
        p = self.params
        return [
            "--model", self.config.model, "--weights", self.weights,
            "--steps", str(p.steps), "--cfg-scale", str(p.cfg_scale),
            "--max-seconds", str(p.max_seconds),
            "--default-seconds", str(min(p.default_seconds, p.max_seconds)),
        ]  # fmt: skip


def hydrate(result: dict[str, Any]) -> MusicResult:
    """Child result dict → MusicResult; an error envelope stays an error."""
    rid = str(result.get("id", "unknown"))
    if result.get("error"):
        return MusicResult(id=rid, error=str(result["error"]))
    return MusicResult.model_validate({**result, "id": rid})
