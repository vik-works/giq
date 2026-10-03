# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Depth worker: one RGB image in, one depth map out (Depth Anything V2).

The child (``_depth_child``) runs the model and returns, per task, a 16-bit
PNG of the prediction at the input's resolution plus the prediction's range,
so the parent has nothing to assemble — it only hydrates the dicts into
``DepthResult``. Everything about the map's meaning is documented on that
model.

Why this model and not Marigold V2: Marigold V2 is a
LoRA on Qwen-Image-Edit-2509, a 20B DiT run NF4 — 17 GB of VRAM at 1024²,
~40 GiB of weights to fetch, and a diffusers/bitsandbytes/peft stack giq
does not carry. It resolves hair and foliage edges this model blurs, and it
is the one to reach for if that matters. Depth Anything V2 is native to the
transformers giq already runs, weighs 0.1-1.3 GB, and answers in
milliseconds — so it can sit beside the residents rather than evict them.

Sleepy models (load on demand, evictable), though the small one is cheap
enough to pin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from giq.adapters._subprocess import SubprocessAdapter
from giq.models import DepthResult
from giq.registry import vram_for
from giq.weights import require_path

# Upper bound for an unregistered model; the registry has the measured ones.
DEPTH_VRAM_GB = 2.0


@dataclass
class DepthConfig:
    model: str = "depth-anything-v2-small"


class DepthAdapter(SubprocessAdapter):
    """Depth Anything V2 in a child process."""

    child_module: ClassVar[str] = "giq_depth._depth_child"
    modality: ClassVar[str] = "depth"

    def __init__(self, config: DepthConfig, device: str | None = None):
        super().__init__(config, device)
        # The recipe's weights.path, under GIQ_DEPTH_MODELS_DIR when that is
        # set. Unknown model or no weights: fail here, not at spawn.
        self.weights = require_path(config.model)

    def child_args(self) -> list[str]:
        return ["--model", self.config.model, "--weights", self.weights]

    @property
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=DEPTH_VRAM_GB)

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[DepthResult]:
        raw_results = await super().run_batch(tasks, params)
        return [hydrate(r) for r in raw_results]


def hydrate(result: dict[str, Any]) -> DepthResult:
    """Child result dict → DepthResult; an error envelope stays an error."""
    rid = str(result.get("id", "unknown"))
    if result.get("error"):
        return DepthResult(id=rid, error=str(result["error"]))
    return DepthResult.model_validate({**result, "id": rid})
