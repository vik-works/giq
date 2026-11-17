# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""SkinTokens' recipe parameters: the schema a skintokens recipe's `params`
is validated against. Registered with the engine (giq_rig.plugin)."""

from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import field_validator

from giq.recipes.schema import EngineParams

# The checkpoint SkinTokens recommends: TokenRig trained on ArticulationXL 2.0,
# VRoid Hub and ModelsResource, then refined with GRPO.
CHECKPOINT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"


class SkinTokensParams(EngineParams):
    """What a skintokens recipe may set. See docs/engines.md."""

    # The TokenRig checkpoint, relative to the recipe's weights.path. The
    # skin tokenizer it was trained with is named inside it, relative to the
    # same root, so one snapshot of the repository serves both.
    checkpoint: str = CHECKPOINT

    @field_validator("checkpoint")
    @classmethod
    def _inside_the_weights(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError("`checkpoint` is a path inside weights.path, without '..'")
        return value
