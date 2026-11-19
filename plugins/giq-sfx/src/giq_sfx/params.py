# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""MOSS-SoundEffect's recipe parameters: the schema a moss-sfx recipe's
`params` is validated against. Registered with the engine (giq_sfx.plugin)."""

from __future__ import annotations

from pydantic import Field

from giq.recipes.schema import EngineParams

# How long a start may take with the DiT compiled: the warm-up render that
# compiles it takes minutes on a cold inductor cache.
COMPILE_READY_TIMEOUT = 1200.0
# Without it the child loads 11 GB of weights and is ready.
READY_TIMEOUT = 300.0


class MossSfxParams(EngineParams):
    """What a moss-sfx recipe may set. See docs/engines.md."""

    # Flow-matching solver steps a task gets unless it asks for others;
    # MOSS-SoundEffect's own default is 100.
    steps: int = Field(default=100, ge=1, le=200)
    # Classifier-free guidance a task gets unless it asks for another: how
    # strictly the sound follows the prompt. 1 turns it off and halves the
    # DiT passes.
    guidance: float = Field(default=4.0, ge=1.0, le=20.0)
    # The flow-match schedule's shift (MOSS's sigma_shift): higher spends
    # more of the steps on the noisy end, on the sound's overall shape.
    shift: float = Field(default=5.0, gt=0.0, le=20.0)
    # torch.compile with CUDA graphs over the DiT, as upstream ships it.
    # Off by default: the model loads on demand and is evicted, and every
    # load would pay a compile of minutes (tens of seconds from a warm
    # inductor cache) before its first task, and the graphs hold memory the
    # VRAM figure does not count. On, the child compiles at load with a
    # short warm-up render, so no task waits on it.
    compile: bool = False

    @property
    def ready_timeout(self) -> float:
        return COMPILE_READY_TIMEOUT if self.compile else READY_TIMEOUT
