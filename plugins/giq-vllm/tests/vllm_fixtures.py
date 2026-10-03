# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the vllm tests: a fake checkpoint and recipe documents."""

import json
from pathlib import Path

from giq.recipes.schema import Recipe
from giq_vllm.params import VllmParams

NAME = "qwen3.8-27b-nvfp4"
BUDGET = {"kv_cache_memory": "6G", "max_model_len": 131072}


def make_checkpoint(root: Path, mtp: int | None = 1, nested: bool = True) -> Path:
    weights = root / "nvidia-Qwen3.8-27B-NVFP4"
    weights.mkdir(parents=True, exist_ok=True)
    text = {"num_hidden_layers": 64}
    if mtp is not None:
        text["mtp_num_hidden_layers"] = mtp
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "vision_config": {}}
    if nested:
        config["text_config"] = text
    else:
        config.update(text)
    (weights / "config.json").write_text(json.dumps(config))
    return weights


def doc(
    weights: Path | str, profile: str | None = None, vram: dict | None = None, **params
) -> dict:
    out = {
        "name": NAME,
        "worker": "llm",
        "engine": "vllm",
        "weights": {"path": str(weights), "format": "modelopt"},
        "capabilities": ["chat", "vision"],
        "params": {**BUDGET, **params},
        "vram": vram or {"weights_gb": 19.92, "overhead_gb": 3.5},
    }
    if profile is not None:
        out["profile"] = profile
    return out


def make_recipe(weights: Path, profile: str | None = None, **params) -> Recipe:
    return Recipe.model_validate(doc(weights, profile, **params))


def params_of(profile: str | None = None, **given) -> VllmParams:
    p = Recipe.model_validate(doc("/nonexistent/ckpt", profile, **given)).params
    assert isinstance(p, VllmParams)
    return p
