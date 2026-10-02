# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""What a recipe file may say (ADR-002; the terms are ADR-003's).

A recipe is one servable model: a name clients send, the weights it runs,
the engine that runs them, that engine's parameters, residency defaults and
the VRAM figure the scheduler gates on. The file format covers the whole
design; this schema accepts only what giq acts on. A key it does not know is
an error, and so is a known key giq does not honour yet (``profile`` on an
engine without profiles, ``residency.gpu``, ``default_policy: off``) — a file
must never claim a behaviour the service does not have.

Every model is strict and frozen: a typo is refused at load rather than
silently becoming a default, and a loaded recipe cannot be edited in place,
so a snapshot handed out stays what it was when it was validated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    ValidationError,
    field_validator,
    model_validator,
)

from giq.engines import ENGINE_ALIASES, ENGINE_OF_BACKEND, canonical_engine
from giq.models import Modality

logger = logging.getLogger(__name__)

# Names are what clients send and what log lines, file names and the
# /control routes carry, so they stay path- and flag-safe: no slash, no
# leading dash or dot.
Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=128)]

# A value that ends up as one argv element. It may not start with a dash, so
# it can never be read as a flag of its own.
Arg = Annotated[str, Field(min_length=1, pattern=r"^[^-]")]

# Which engines can serve which worker. The workers are written against one
# runtime each (ocr against two), so a recipe that pairs a
# worker with a foreign engine could never load.
MODALITY_ENGINES: dict[str, frozenset[str]] = {
    "llm": frozenset({"llama.cpp", "vllm"}),
    "text2image": frozenset({"sd.cpp"}),
    "image_edit": frozenset({"sd.cpp"}),
    "tts": frozenset({"kokoro"}),
    "stt": frozenset({"faster-whisper"}),
    "audio": frozenset({"faster-whisper+pyannote"}),
    "embed": frozenset({"speechbrain"}),
    "ocr": frozenset({"transformers", "transformers-4.57"}),
    "depth": frozenset({"transformers"}),
}

# llama.cpp's KV cache types (`--cache-type-k/-v`).
KvCacheType = Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]

Capability = Literal["chat", "vision"]

# Concurrent jobs allowed on a resident's lane, by modality. llm matches
# llama-server's 4 slots, so a client fanning out a few chat calls at once
# gets them served in parallel; embed takes 2; audio is GPU-heavy and serial.
DEFAULT_LANE_WIDTH: dict[str, int] = {"llm": 4, "audio": 1, "embed": 2}

# Request-body values: the sampler fields llama-server reads are scalars.
RequestValue = bool | int | float | str


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# A named file or snapshot a model needs besides (or instead of) its main
# weights: an image model's diffusion model, text encoder and VAE, OCR's
# layout model. Lower-case identifiers, so a part name reads the same in a
# file, a log line and an error.
PartName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]

# The parts each modality's adapter reads. A part it does not know would be a
# file written down and never loaded, so it is refused like an unknown key.
MODALITY_PARTS: dict[str, frozenset[str]] = {
    "text2image": frozenset({"diffusion", "text_encoder", "vae", "lora"}),
    "image_edit": frozenset({"diffusion", "text_encoder", "vae", "lora"}),
    "ocr": frozenset({"layout"}),
    "audio": frozenset({"asr", "diarization"}),
}


class WeightsPart(_Strict):
    """One file or snapshot of a multi-part model, with its own provenance.

    A part pulled from another repository than the main weights records
    where it came from, because its licence may differ — FLUX.2's VAE is
    Apache-2.0 from the klein repository and non-commercial from [dev].
    """

    # Resolved like Weights.path.
    path: str | None = None
    source: str | None = None
    revision: str | None = None
    format: Literal["gguf", "safetensors", "modelopt"] | None = None
    licence: str | None = None

    @model_validator(mode="after")
    def _somewhere(self) -> WeightsPart:
        if not (self.path or self.source):
            raise ValueError("a part needs a path or a source")
        return self


class Weights(_Strict):
    """Where the model's files are and what they are."""

    # Relative to giq.paths.models_dir(); `~` or absolute is used as written.
    path: str | None = None
    # Provenance: where the files came from and at which revision. Recorded,
    # not fetched — giq downloads nothing on its own.
    source: str | None = None
    revision: str | None = None
    format: Literal["gguf", "safetensors", "modelopt"] | None = None
    licence: str | None = None
    # The other files the model needs, by the name its worker reads them
    # under. A bare string is the part's path.
    parts: dict[PartName, WeightsPart] = Field(default_factory=dict)

    @field_validator("parts", mode="before")
    @classmethod
    def _path_shorthand(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: {"path": p} if isinstance(p, str) else p for k, p in v.items()}
        return v


class Vram(_Strict):
    """The figure the VRAM gate and the eviction planner use."""

    gb: float = Field(gt=0)
    # vllm with a KV budget in bytes: the figure is the loaded weights plus
    # that budget plus everything else the process holds (CUDA context,
    # activations, graphs), the two measured once per checkpoint and engine
    # version. With both set, `gb` may be left out and is derived; if given,
    # it must agree.
    weights_gb: float | None = Field(default=None, gt=0)
    overhead_gb: float | None = Field(default=None, ge=0)
    # True only for a figure observed on real hardware; otherwise the number
    # is an upper-bound estimate and the catalog says so.
    measured: bool = False
    measured_on: str | None = None
    measured_with: dict[str, str | int | float] | None = None
    notes: str | None = None


class Residency(_Strict):
    """Residency defaults; operator overrides live in the model_policy table."""

    # Position in the default resident set, lowest first. Unset = loads on
    # demand and is evictable.
    priority: int | None = Field(default=None, ge=0)
    # The policy the priority implies (pinned with one, auto without). It may
    # be written out, and must then agree.
    default_policy: Literal["pinned", "auto", "off"] | None = None
    # Placement comes from config.yaml `gpu.bind` and the dashboard for now.
    gpu: str | None = None

    @model_validator(mode="after")
    def _honoured_only(self) -> Residency:
        if self.gpu is not None:
            raise ValueError(
                "residency.gpu is not supported yet; bind the model with config.yaml "
                "gpu.bind or the dashboard"
            )
        if self.default_policy == "off":
            raise ValueError("residency.default_policy: off is not supported yet")
        implied = "pinned" if self.priority is not None else "auto"
        if self.default_policy is not None and self.default_policy != implied:
            raise ValueError(
                f"residency.default_policy {self.default_policy!r} disagrees with "
                f"priority {self.priority!r} (which means {implied!r})"
            )
        return self


class EngineParams(_Strict):
    """Parameters of an engine that takes none from a recipe."""

    def given(self) -> dict[str, Any]:
        """Only the parameters the file set — unset ones take the engine default."""
        return {k: getattr(self, k) for k in self.model_fields_set}


class LlamaCppParams(EngineParams):
    """llama-server launch parameters. Unset = the default in adapters/llama_cpp.py."""

    ctx_size: int | None = Field(default=None, ge=512)
    cache_type_k: KvCacheType | None = None
    cache_type_v: KvCacheType | None = None
    # "on"/"off"/"auto" go to --reasoning; "template" omits the flag and lets
    # the chat template decide.
    reasoning: Literal["on", "off", "auto", "template"] | None = None
    reasoning_budget: int | None = Field(default=None, ge=0)
    parallel: int | None = Field(default=None, ge=1)
    spec_type: Arg | None = None
    # The vision projector, resolved like weights.path.
    mmproj: Arg | None = None
    # --alias: the id llama-server reports on /v1/models.
    alias: Arg | None = None
    # giq's in-flight loop guard on the thinking channel.
    loop_guard: bool | None = None
    # Seconds a start may take before it is abandoned; unset = the engine's
    # default (adapters/llama_cpp.py DEFAULT_READY_TIMEOUT). A big GGUF on a slow
    # disk is the reason to raise it.
    ready_timeout: float | None = Field(default=None, ge=10, le=3600)

    @model_validator(mode="after")
    def _kv_pair(self) -> LlamaCppParams:
        # Mixed K/V types fall off the fused attention kernel (see
        # DEFAULT_CACHE_TYPE_K in adapters/llama_cpp.py), so they are set as a pair.
        if (self.cache_type_k is None) != (self.cache_type_v is None):
            raise ValueError("cache_type_k and cache_type_v are set together")
        if self.cache_type_k != self.cache_type_v:
            raise ValueError(
                f"cache_type_k {self.cache_type_k} and cache_type_v {self.cache_type_v} "
                "must match; a mixed pair loses the fused attention kernel"
            )
        return self


# --- vllm ---------------------------------------------------------------------

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


ENGINE_PARAMS: dict[str, type[EngineParams]] = {"llama.cpp": LlamaCppParams, "vllm": VllmParams}


class Recipe(_Strict):
    """One servable model, as a recipe file declares it."""

    name: Name
    # The kinds of job this recipe serves (ADR-003). Usually one; flux_klein
    # serves text2image and image_edit from one sd-server.
    modalities: tuple[str, ...] = Field(min_length=1)
    engine: str
    label: str = ""
    detail: str = ""
    weights: Weights | None = None
    capabilities: tuple[Capability, ...] = ()
    # A named parameter set of the engine (ADR-002 D9).
    profile: str | None = None
    # Serialised as the engine's own schema, not the empty base.
    params: SerializeAsAny[EngineParams] = EngineParams()
    # Body fields giq puts under the caller's request; the caller's win.
    request_defaults: dict[str, RequestValue] = Field(default_factory=dict)
    residency: Residency = Residency()
    vram: Vram
    # Other names clients may send for this recipe.
    aliases: tuple[Name, ...] = ()
    # Concurrent jobs on a resident's lane; unset = the modality's default.
    lane_width: int | None = Field(default=None, ge=1)
    # Tasks per job: one figure, or one per modality where they differ (an
    # edit carries a reference image a render does not).
    max_batch: Annotated[int, Field(ge=1)] | dict[str, Annotated[int, Field(ge=1)]] | None = None
    voices: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _old_worker_key(cls, data: Any) -> Any:
        # Before ADR-003 a recipe named one `worker`; an operator's file
        # written then still loads, as a recipe serving that one modality.
        if isinstance(data, dict) and "worker" in data and "modalities" not in data:
            worker = data["worker"]
            logger.warning(
                f"recipe {data.get('name')!r}: `worker: {worker}` is now "
                f"`modalities: [{worker}]` (ADR-003)"
            )
            data = {k: v for k, v in data.items() if k != "worker"}
            data["modalities"] = [worker]
        return data

    @field_validator("modalities")
    @classmethod
    def _known_modalities(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        known = {m.value for m in Modality}
        if unknown := [m for m in v if m not in known]:
            raise ValueError(
                f"unknown modality {', '.join(map(repr, unknown))} "
                f"(known: {', '.join(sorted(known))})"
            )
        if len(set(v)) != len(v):
            raise ValueError("modalities repeat")
        return tuple(Modality(m).value for m in v)

    @field_validator("engine")
    @classmethod
    def _known_engine(cls, v: str) -> str:
        v = canonical_engine(v, "recipe file")
        if v not in ENGINE_OF_BACKEND:
            raise ValueError(f"unknown engine {v!r} (known: {', '.join(ENGINE_OF_BACKEND)})")
        return v

    @model_validator(mode="before")
    @classmethod
    def _engine_params(cls, data: Any) -> Any:
        # The parameter schema depends on the engine, so it is chosen here
        # rather than by a union that would accept whichever variant fits.
        if not isinstance(data, dict):
            return data
        params = data.get("params")
        if isinstance(params, EngineParams):
            return data
        engine = str(data.get("engine"))
        # A profile is a set of engine defaults under the file's own params;
        # an unknown one is left for _consistent to report.
        profile = ENGINE_PROFILES.get(ENGINE_ALIASES.get(engine, engine), {}).get(
            str(data.get("profile"))
        )
        if profile is not None:
            params = {**profile, **(params or {})}
        schema = ENGINE_PARAMS.get(ENGINE_ALIASES.get(engine, engine), EngineParams)
        try:
            validated = schema.model_validate(params if params is not None else {})
        except ValidationError as e:
            # Re-raised under `params` so the message says where the key was.
            raise ValidationError.from_exception_data(
                cls.__name__,
                [
                    {
                        "type": err["type"],
                        "loc": ("params", *err["loc"]),
                        "input": err["input"],
                        **({"ctx": err["ctx"]} if "ctx" in err else {}),
                    }
                    for err in e.errors()
                ],
            ) from None
        data = {**data, "params": validated}
        vram = data.get("vram")
        if (
            isinstance(validated, VllmParams)
            and isinstance(vram, dict)
            and "gb" not in vram
            and validated.kv_cache_memory_bytes is not None
            and isinstance(vram.get("weights_gb"), int | float)
            and isinstance(vram.get("overhead_gb"), int | float)
        ):
            data["vram"] = {**vram, "gb": _derived_vram(vram, validated)}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> Recipe:
        for modality in self.modalities:
            engines = MODALITY_ENGINES.get(modality, frozenset())
            if self.engine not in engines:
                raise ValueError(
                    f"engine {self.engine!r} cannot serve modality {modality!r} "
                    f"(it takes: {', '.join(sorted(engines))})"
                )
        if isinstance(self.max_batch, dict):
            if stray := sorted(set(self.max_batch) - set(self.modalities)):
                raise ValueError(
                    f"max_batch names {', '.join(stray)}, which this recipe does not serve"
                )
        if self.profile is not None:
            profiles = ENGINE_PROFILES.get(self.engine)
            if not profiles:
                raise ValueError(f"engine {self.engine!r} has no profiles; set params directly")
            if self.profile not in profiles:
                raise ValueError(
                    f"unknown {self.engine} profile {self.profile!r} (known: {', '.join(profiles)})"
                )
        if self.engine != "vllm" and (
            self.vram.weights_gb is not None or self.vram.overhead_gb is not None
        ):
            raise ValueError("vram.weights_gb and vram.overhead_gb are for engine vllm")
        if self.weights is not None and self.weights.parts:
            readable = frozenset().union(
                *(MODALITY_PARTS.get(m, frozenset()) for m in self.modalities)
            )
            if unknown := sorted(set(self.weights.parts) - readable):
                raise ValueError(
                    f"weights.parts {', '.join(unknown)}: {', '.join(self.modalities)} reads "
                    + (f"only {', '.join(sorted(readable))}" if readable else "no parts")
                )
        if self.name in self.aliases:
            raise ValueError(f"alias {self.name!r} repeats the recipe name")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("aliases repeat")
        if self.engine == "llama.cpp":
            params = self.params
            assert isinstance(params, LlamaCppParams)
            if not (self.weights and self.weights.path):
                raise ValueError("a llama.cpp recipe needs weights.path")
            # Without the projector the weights serve text and silently drop
            # image parts, so vision and mmproj are declared together.
            if ("vision" in self.capabilities) != bool(params.mmproj):
                raise ValueError("capability `vision` and params.mmproj go together")
        elif self.engine == "vllm":
            self._vllm_consistent()
        elif self.request_defaults:
            raise ValueError(f"request_defaults are not supported for engine {self.engine!r}")
        return self

    def _vllm_consistent(self) -> None:
        params = self.params
        assert isinstance(params, VllmParams)
        if not (self.weights and self.weights.path):
            raise ValueError("a vllm recipe needs weights.path (a checkpoint directory)")
        # D4 accepts(): vllm reads Hugging Face checkpoint directories.
        if self.weights.format not in ("safetensors", "modelopt"):
            raise ValueError("a vllm recipe needs weights.format safetensors or modelopt")
        if self.lane_width is not None:
            raise ValueError("lane_width is derived from params.max_num_seqs for vllm (D8)")
        stray = sorted(set(self.request_defaults) - VLLM_REQUEST_DEFAULTS)
        if stray:
            raise ValueError(
                f"request_defaults {stray} are not vllm sampling fields "
                f"(allowed: {', '.join(sorted(VLLM_REQUEST_DEFAULTS))})"
            )
        kv = params.kv_cache_memory_bytes
        if kv is None:
            if self.vram.weights_gb is not None or self.vram.overhead_gb is not None:
                raise ValueError(
                    "vram.weights_gb/overhead_gb need a kv_cache_memory budget; with "
                    "gpu_memory_utilization give vram.gb"
                )
        elif self.vram.weights_gb is not None and self.vram.overhead_gb is not None:
            expected = _derived_vram(self.vram.model_dump(), params)
            if abs(self.vram.gb - expected) > 0.05:
                raise ValueError(
                    f"vram.gb {self.vram.gb} disagrees with weights_gb + kv_cache_memory + "
                    f"overhead_gb = {expected}"
                )
        # D9: what the weights can do. Checked when the checkpoint is on this
        # machine; the worker checks again before it starts the server.
        if params.speculative is not None and params.speculative.method == "mtp":
            from giq.paths import model_path

            hf_config = read_hf_config(model_path(self.weights.path))
            if hf_config is not None and mtp_layers(hf_config) == 0:
                raise ValueError(
                    "params.speculative mtp needs an MTP head, and this checkpoint's "
                    "config.json declares none"
                )

    # --- what the catalog, the scheduler and the dashboard read -----------

    @property
    def modality(self) -> str:
        """The first modality: the one that picks the adapter and the defaults.

        A recipe serving several (flux_klein) runs one process for all of
        them, so any one of them would pick the same adapter.
        """
        return self.modalities[0]

    def serves(self, modality: str) -> bool:
        return str(modality) in self.modalities

    def max_batch_for(self, modality: str) -> int | None:
        if isinstance(self.max_batch, dict):
            return self.max_batch.get(str(modality))
        return self.max_batch

    @property
    def vram_gb(self) -> float:
        return self.vram.gb

    @property
    def measured(self) -> bool:
        """The VRAM figure was observed on real hardware, not estimated."""
        return self.vram.measured

    @property
    def vision(self) -> bool:
        """The recipe takes images alongside text: a capability of an LLM recipe."""
        return "vision" in self.capabilities

    @property
    def mmproj(self) -> str | None:
        """llama.cpp's projector file, without which the weights are text-only."""
        return self.params.mmproj if isinstance(self.params, LlamaCppParams) else None

    @property
    def lanes(self) -> int:
        """Concurrent jobs on this recipe's resident lane.

        vllm's max_num_seqs is how many requests it really runs at once (D8);
        llama.cpp keeps its separate lane width until its -np is aligned.
        """
        if isinstance(self.params, VllmParams):
            return self.params.max_num_seqs
        if self.lane_width is not None:
            return self.lane_width
        return DEFAULT_LANE_WIDTH.get(self.modality, 1)

    @property
    def display(self) -> str:
        return self.label or self.name


def _derived_vram(vram: dict[str, Any], params: VllmParams) -> float:
    kv = params.kv_cache_memory_bytes or 0
    return round(float(vram["weights_gb"]) + kv / 2**30 + float(vram["overhead_gb"]), 2)
