# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""vllm's recipe parameters (ADR-002 D2, D9): the schema a vllm recipe's
`params` is validated against, its named profiles, and the rules a whole
vllm recipe must meet. Registered with the engine (giq_vllm.plugin)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from giq.recipes.schema import Arg, EngineParams, Recipe
from giq.recipes.schema import StrictModel as _Strict

# vllm's KV cache element types (`--kv-cache-dtype`).
VllmKvCacheType = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]

# A byte size as systemd and vllm spell it: binary suffixes (8G = 8 GiB) or a
# plain byte count.
_SIZE_PATTERN = r"^[0-9]+((\.[0-9]+)?[KMGT])?$"
Size = Annotated[str, Field(pattern=_SIZE_PATTERN)]
_SIZE_FACTOR = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}

# vllm's names for its parsers are plain identifiers.
Ident = Annotated[str, Field(pattern=r"^[a-z0-9_]+$", max_length=64)]


def size_bytes(value: str | int) -> int:
    """Bytes from "8G" / "512M" (binary units) or a plain count."""
    if isinstance(value, int):
        return value
    text = value.strip()
    if text[-1] in _SIZE_FACTOR:
        return int(float(text[:-1]) * _SIZE_FACTOR[text[-1]])
    return int(text)


class Speculative(_Strict):
    """Speculative decoding, named abstractly (D9); the adapter spells the flag."""

    # mtp drafts with the checkpoint's own multi-token-prediction head.
    method: Literal["mtp"]
    tokens: int = Field(ge=1, le=8)


class StructuredOutputs(_Strict):
    """How constrained decoding builds the grammar for a json_schema request.

    The default matters: vllm's own default (`backend: auto`,
    `disable_any_whitespace: false`) lets the grammar emit arbitrary whitespace
    between every JSON token, and on a large or deep schema the model walks into
    an unbounded run of newlines and spaces that never closes the object — the
    reply fills to `max_tokens` as 90%-whitespace invalid JSON. Pinning an
    explicit grammar backend and forbidding the free whitespace makes the same
    schema converge. giq therefore defaults a vllm instance to the convergent
    setting rather than vllm's; a recipe can still ask for the permissive one.
    """

    # auto lets vllm choose; disable_any_whitespace below is only honoured by
    # the two grammar backends, so the default names one.
    backend: Literal["auto", "xgrammar", "guidance"] = "xgrammar"
    disable_any_whitespace: bool = True

    @model_validator(mode="after")
    def _whitespace_needs_a_grammar_backend(self) -> StructuredOutputs:
        # vllm itself rejects the pair, minutes into a start; catch it when the
        # recipe loads instead.
        if self.disable_any_whitespace and self.backend == "auto":
            raise ValueError(
                "disable_any_whitespace needs backend 'xgrammar' or 'guidance', not 'auto'"
            )
        return self


class VllmParams(EngineParams):
    """`vllm serve` launch parameters. See docs/engines.md for each."""

    # The budget — exactly one of the two. A KV size is the same on any card,
    # so co-resident models keep their room; a fraction means 29 GB on a 32 GB
    # card and 89 GB on a 96 GB one.
    kv_cache_memory: Size | int | None = None
    gpu_memory_utilization: float | None = Field(default=None, ge=0.05, le=0.98)
    # Required: with the checkpoint's native window the KV pool may not hold
    # one full request, and the start fails minutes in.
    max_model_len: int = Field(ge=256, le=1_048_576)
    kv_cache_dtype: VllmKvCacheType = "auto"
    # The most requests scheduled at once; giq dispatches exactly this many.
    max_num_seqs: int = Field(default=16, ge=1, le=1024)
    max_num_batched_tokens: int | None = Field(default=None, ge=256)
    speculative: Speculative | None = None
    enforce_eager: bool = False
    reasoning_parser: Ident | None = None
    tool_call_parser: Ident | None = None
    # A Jinja chat template file, resolved like weights.path (relative to
    # GIQ_MODELS_DIR, `~`/absolute as written). Replaces the checkpoint's own
    # chat_template.jinja. Unset = the checkpoint's template.
    chat_template_file: Arg | None = None
    # Constrained decoding for json_schema / structured_outputs requests. The
    # default (xgrammar, no free whitespace) keeps a large schema from
    # diverging into a whitespace run; see StructuredOutputs.
    structured_outputs: StructuredOutputs = Field(default_factory=StructuredOutputs)
    # RAM ceiling on the engine process (systemd MemoryMax); null = none.
    # FlashInfer's first-use kernel builds run two jobs at up to ~15 GB each
    # (workers/vllm.py DEFAULT_MEMORY_MAX).
    memory_max: Annotated[str, Field(pattern=r"^[1-9][0-9]*[KMGT]?$")] | None = "40G"
    # Seconds a start may take. Measured on an RTX 5090: 192 s cold (64 of
    # them CUDA graph capture), 82 s warm; ten minutes covers a slower disk
    # without letting a wedged start hold the runner's lock for ever.
    ready_timeout: float = Field(default=600.0, ge=30, le=3600)

    @model_validator(mode="after")
    def _one_budget(self) -> VllmParams:
        budgets = (self.kv_cache_memory, self.gpu_memory_utilization)
        if all(b is None for b in budgets):
            raise ValueError(
                "a VRAM budget is required: kv_cache_memory (preferred) or gpu_memory_utilization"
            )
        if all(b is not None for b in budgets):
            raise ValueError("give one budget, kv_cache_memory or gpu_memory_utilization")
        kv = self.kv_cache_memory_bytes
        if kv is not None and kv < 2**28:
            raise ValueError("kv_cache_memory below 256M holds no useful context")
        return self

    @property
    def kv_cache_memory_bytes(self) -> int | None:
        if self.kv_cache_memory is None:
            return None
        return size_bytes(self.kv_cache_memory)


# Named parameter sets (D9), under a recipe's own `params`. The measured
# trade-off on an RTX 5090 with NVIDIA's NVFP4 Qwen3.8-27B: MTP with three
# draft tokens took one request from 72 to 117 tok/s and the total at 16
# parallel requests from 937 down to 750, accepting 0.66 / 0.37 / 0.17 by
# draft position — so interactive drafts two, with a larger scheduling
# budget against the TTFT MTP added under load, and throughput drafts none.
# 16 parallel was the measured point; 32 is only a ceiling over the paged KV
# pool (short requests fill it, long ones wait inside the budget), so it
# suits a large card without costing a small one memory. The VRAM budget is
# in neither: it belongs to the recipe and the card it shares.
ENGINE_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "vllm": {
        "interactive": {
            "speculative": {"method": "mtp", "tokens": 2},
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "kv_cache_dtype": "fp8",
        },
        "throughput": {
            "speculative": None,
            "max_num_seqs": 32,
            "kv_cache_dtype": "fp8",
        },
    },
}

# Sampling fields a vllm recipe may default; the caller's body wins.
VLLM_REQUEST_DEFAULTS = frozenset(
    {
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    }
)

# Where Hugging Face configs keep the size of a multi-token-prediction head:
# Qwen3.5+ and MiMo name it mtp_num_hidden_layers, DeepSeek-V3 and GLM
# num_nextn_predict_layers. Multimodal checkpoints nest it in text_config.
_MTP_KEYS = ("mtp_num_hidden_layers", "num_nextn_predict_layers")


def mtp_layers(hf_config: dict[str, Any]) -> int:
    """Size of a checkpoint's MTP head; 0 when it has none."""
    for section in (hf_config, hf_config.get("text_config") or {}):
        for key in _MTP_KEYS:
            value = section.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return 0


def read_hf_config(weights_dir: str | Path) -> dict[str, Any] | None:
    """A checkpoint's config.json; None when it is not on this machine."""
    path = Path(weights_dir) / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def vllm_consistent(recipe: Recipe) -> None:
    """What a vllm recipe must say besides the schema."""
    params = recipe.params
    assert isinstance(params, VllmParams)
    if not (recipe.weights and recipe.weights.path):
        raise ValueError("a vllm recipe needs weights.path (a checkpoint directory)")
    # D4 accepts(): vllm reads Hugging Face checkpoint directories.
    if recipe.weights.format not in ("safetensors", "modelopt"):
        raise ValueError("a vllm recipe needs weights.format safetensors or modelopt")
    if recipe.weights.parts:
        # The checkpoint directory carries everything vllm loads, a vision
        # tower included.
        raise ValueError("a vllm recipe reads one checkpoint directory; it takes no parts")
    if recipe.lane_width is not None:
        raise ValueError("lane_width is derived from params.max_num_seqs for vllm (D8)")
    kv = params.kv_cache_memory_bytes
    if kv is None:
        if recipe.vram.weights_gb is not None or recipe.vram.overhead_gb is not None:
            raise ValueError(
                "vram.weights_gb/overhead_gb need a kv_cache_memory budget; with "
                "gpu_memory_utilization give vram.gb"
            )
    elif recipe.vram.weights_gb is not None and recipe.vram.overhead_gb is not None:
        expected = derived_vram(recipe.vram.model_dump(), params)
        if expected is not None and abs(recipe.vram.gb - expected) > 0.05:
            raise ValueError(
                f"vram.gb {recipe.vram.gb} disagrees with weights_gb + kv_cache_memory + "
                f"overhead_gb = {expected}"
            )
    # D9: what the weights can do. Checked when the checkpoint is on this
    # machine; the worker checks again before it starts the server.
    if params.speculative is not None and params.speculative.method == "mtp":
        from giq.paths import model_path

        hf_config = read_hf_config(model_path(recipe.weights.path))
        if hf_config is not None and mtp_layers(hf_config) == 0:
            raise ValueError(
                "params.speculative mtp needs an MTP head, and this checkpoint's "
                "config.json declares none"
            )


def derived_vram(vram: Mapping[str, Any], params: VllmParams) -> float | None:
    """vram.gb from its parts: weights + the KV budget + overhead.

    None without a KV budget in bytes: with a fraction of the card the
    total is not knowable from the recipe, so vram.gb has to be given."""
    if params.kv_cache_memory_bytes is None:
        return None
    kv = params.kv_cache_memory_bytes
    return round(float(vram["weights_gb"]) + kv / 2**30 + float(vram["overhead_gb"]), 2)
