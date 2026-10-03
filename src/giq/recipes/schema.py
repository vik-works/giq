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

import logging
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

from giq import plugins
from giq.engines import canonical_engine

logger = logging.getLogger(__name__)

# Names are what clients send and what log lines, file names and the
# /control routes carry, so they stay path- and flag-safe: no slash, no
# leading dash or dot.
Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=128)]

# A value that ends up as one argv element. It may not start with a dash, so
# it can never be read as a flag of its own.
Arg = Annotated[str, Field(min_length=1, pattern=r"^[^-]")]

# llama.cpp's KV cache types (`--cache-type-k/-v`).
KvCacheType = Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]

Capability = Literal["chat", "vision"]

# Request-body values: the sampler fields llama-server reads are scalars.
RequestValue = bool | int | float | str


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# For an engine's own nested parameter models (a plugin's): strict and
# frozen like everything else in a recipe.
StrictModel = _Strict


# A named file or snapshot a model needs besides (or instead of) its main
# weights: an image model's diffusion model, text encoder and VAE, OCR's
# layout model. Lower-case identifiers, so a part name reads the same in a
# file, a log line and an error.
PartName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]


def _source_file(source: str | None) -> str | None:
    """The file in the repository an ``hf:org/repo/file`` source names, or None."""
    if source is None or not source.startswith("hf:"):
        return None
    return source[3:].split("/", 2)[2] if source.count("/") >= 2 else None


def _fetchable(path: str | None, source: str | None) -> None:
    """A source giq can fetch: ``hf:org/repo`` or ``hf:org/repo/file``.

    One file has nowhere to go but the path it is loaded from; a whole
    repository without a path goes to the Hugging Face cache.
    """
    if source is None:
        return
    if source.startswith("hf:") and source[3:].count("/") < 1:
        raise ValueError(f"source {source!r} names no repository: hf:org/repo[/file]")
    if _source_file(source) and not path:
        raise ValueError(f"source {source!r} names one file; give the path it goes to")


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
        _fetchable(self.path, self.source)
        return self


class Weights(_Strict):
    """Where the model's files are and what they are."""

    # Relative to giq.paths.models_dir(); `~` or absolute is used as written.
    path: str | None = None
    # Where the files come from and at which revision (ADR-005): `hf:org/repo`
    # for a whole repository, `hf:org/repo/path/in/repo` for one file of it.
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

    @model_validator(mode="after")
    def _source(self) -> Weights:
        _fetchable(self.path, self.source)
        return self


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
    # A Jinja chat template file, resolved like weights.path (relative to
    # GIQ_MODELS_DIR, `~`/absolute as written). Replaces the GGUF's embedded
    # template — a fixed upstream Qwen file, say. Unset = the model's own.
    chat_template_file: Arg | None = None
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


def _projector_part(data: dict[str, Any]) -> dict[str, Any]:
    """``params.mmproj`` read as ``weights.parts.mmproj`` (ADR-005 D2).

    The projector is weights: inventoried, sized, fetched and deleted with
    the rest. A file that still names it as a parameter loads, with a warning.
    """
    params = data.get("params")
    if not (isinstance(params, dict) and "mmproj" in params):
        return data
    logger.warning(
        f"recipe {data.get('name')!r}: `params.mmproj` is now `weights.parts.mmproj` (ADR-005)"
    )
    weights = dict(data.get("weights") or {})
    parts = dict(weights.get("parts") or {})
    if "mmproj" in parts:
        raise ValueError("the projector is given twice: params.mmproj and weights.parts.mmproj")
    parts["mmproj"] = params["mmproj"]
    return {
        **data,
        "params": {k: v for k, v in params.items() if k != "mmproj"},
        "weights": {**weights, "parts": parts},
    }


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
        known = plugins.modalities()
        if unknown := [m for m in v if m not in known]:
            raise ValueError(
                f"unknown modality {', '.join(map(repr, unknown))} "
                f"(known: {', '.join(sorted(known))}; a plugin adds others)"
            )
        if len(set(v)) != len(v):
            raise ValueError("modalities repeat")
        return tuple(str(m) for m in v)

    @field_validator("engine")
    @classmethod
    def _known_engine(cls, v: str) -> str:
        v = canonical_engine(v, "recipe file")
        if plugins.engine(v) is None:
            known = ", ".join(sorted(plugins.engines()))
            raise ValueError(f"unknown engine {v!r} (known: {known}; a plugin adds others)")
        return v

    @model_validator(mode="before")
    @classmethod
    def _engine_params(cls, data: Any) -> Any:
        # The parameter schema depends on the engine, so it is chosen here
        # rather than by a union that would accept whichever variant fits.
        if not isinstance(data, dict):
            return data
        data = _projector_part(data)
        params = data.get("params")
        if isinstance(params, EngineParams):
            return data
        engine = plugins.engine(str(data.get("engine")))
        # A profile is a set of engine defaults under the file's own params;
        # an unknown one is left for _consistent to report.
        profile = engine.profiles.get(str(data.get("profile"))) if engine else None
        if profile is not None:
            params = {**profile, **(params or {})}
        schema: type[EngineParams] = (
            engine.params if engine is not None and engine.params is not None else EngineParams
        )
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
            engine is not None
            and engine.derive_vram is not None
            and isinstance(vram, dict)
            and "gb" not in vram
            and isinstance(vram.get("weights_gb"), int | float)
            and isinstance(vram.get("overhead_gb"), int | float)
        ):
            derived = engine.derive_vram(vram, validated)
            if derived is not None:
                data["vram"] = {**vram, "gb": derived}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> Recipe:
        engine = plugins.engine(self.engine)
        assert engine is not None  # _known_engine
        for modality in self.modalities:
            engines = plugins.engines_for(modality)
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
            profiles = engine.profiles
            if not profiles:
                raise ValueError(f"engine {self.engine!r} has no profiles; set params directly")
            if self.profile not in profiles:
                raise ValueError(
                    f"unknown {self.engine} profile {self.profile!r} (known: {', '.join(profiles)})"
                )
        if engine.derive_vram is None and (
            self.vram.weights_gb is not None or self.vram.overhead_gb is not None
        ):
            raise ValueError(
                f"vram.weights_gb and vram.overhead_gb are not supported for engine {self.engine!r}"
            )
        if self.weights is not None and self.weights.parts:
            readable = frozenset().union(
                *(spec.parts for m in self.modalities if (spec := plugins.modality(m)))
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
        allowed = engine.request_defaults
        if allowed is not None and (stray := sorted(set(self.request_defaults) - allowed)):
            raise ValueError(
                f"request_defaults {stray} are not supported for engine {self.engine!r}"
                + (f" (allowed: {', '.join(sorted(allowed))})" if allowed else "")
            )
        if engine.validate is not None:
            engine.validate(self)
        return self

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
        """llama.cpp's projector file, without which the weights are text-only:
        the ``mmproj`` part's path, unresolved."""
        part = self.weights.parts.get("mmproj") if self.weights else None
        return part.path if part is not None else None

    @property
    def lanes(self) -> int:
        """Concurrent jobs on this recipe's resident lane.

        An engine that runs several requests itself says how many from its
        params (vllm's max_num_seqs, D8); otherwise the recipe's lane_width,
        else its modality's default.
        """
        engine = plugins.engine(self.engine)
        if engine is not None and engine.lanes is not None:
            derived = engine.lanes(self.params)
            if derived is not None:
                return derived
        if self.lane_width is not None:
            return self.lane_width
        spec = plugins.modality(self.modality)
        return spec.lane_width if spec is not None else 1

    @property
    def display(self) -> str:
        return self.label or self.name


# --- the engines' own rules, registered with them (giq.builtins) -------------


def llamacpp_consistent(recipe: Recipe) -> None:
    """What a llama.cpp recipe must say besides the schema."""
    params = recipe.params
    assert isinstance(params, LlamaCppParams)
    if not (recipe.weights and recipe.weights.path):
        raise ValueError("a llama.cpp recipe needs weights.path")
    # Without the projector the weights serve text and silently drop image
    # parts, so vision and the projector are declared together.
    if ("vision" in recipe.capabilities) != bool(recipe.mmproj):
        raise ValueError("capability `vision` and weights.parts.mmproj go together")
