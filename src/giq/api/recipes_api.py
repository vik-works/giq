# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The recipes, as an operator manages them (ADR-003).

``GET /recipes`` is the catalog in the domain's own terms: each recipe with
its modalities and engine, whether its weights are installed and whether it
can run here at all (its availability, ADR-005), its residency
and card, whether it fits that card now, and the instance running it if
there is one. The writes set residency and card by recipe name.

Every write is refused with 409 when it would leave the scheduler with a
set it can never satisfy: two LLMs pinned to one card, a pinned set that
does not fit its card (unless ``force``), a recipe bound to a card it can
never fit.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from giq.availability import availability
from giq.engines import runtime_of
from giq.gpus import get_gpus, resolve_device, selected_device
from giq.policy import RESIDENT_SET_HEADROOM_GB, get_policy_store
from giq.recipes.schema import Recipe
from giq.registry import all_recipes, get_recipe, resident_defaults
from giq.runner import get_runner
from giq.storage import installed
from giq.vram import get_vram_status, margin_for, reserve_for
from giq.weights import locations, weights_id

router = APIRouter()


class ResidencyRequest(BaseModel):
    policy: str = Field(description="pinned (kept loaded) | auto (on demand) | off")
    reason: str | None = None
    # Pin even when the card's pinned set would over-commit.
    force: bool = False


class CardRequest(BaseModel):
    # An index or a UUID; null unbinds (the default card).
    device: str | int | None
    force: bool = False


def _recipe_or_404(name: str) -> Recipe:
    recipe = get_recipe(name)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"no recipe {name!r}")
    return recipe


class _Machine:
    """What every entry is judged against, read once per request."""

    def __init__(self, cards: dict, last_used: dict[str, float]) -> None:
        self.last_used = last_used
        runner = get_runner()
        self.store = get_policy_store()
        self.cards = cards
        self.vram = get_vram_status()
        self.defaults = set(resident_defaults())
        self.instances = {i.recipe: i for i in runner.instances()}
        # VRAM ready residents hold on each card: what an eviction could free.
        self.evictable: dict[str | None, float] = {}
        for inst in runner.instances():
            recipe = get_recipe(inst.recipe)
            if inst.residency == "resident" and inst.state == "ready" and recipe:
                self.evictable[inst.device] = self.evictable.get(inst.device, 0.0) + recipe.vram_gb


def _fit(recipe: Recipe, where: str | None, m: _Machine) -> tuple[str, float]:
    """Does ``recipe`` fit its card now, after evictions, or never — and what it needs."""
    needed = recipe.vram_gb + margin_for(recipe.vram_gb) + reserve_for(where)
    inst = m.instances.get(recipe.name)
    if inst is not None and inst.state == "ready":
        return "loaded", needed
    card = m.cards.get(where)
    total = card.vram_total_gb if card else m.vram.total_gb
    free = card.vram_free_gb if card else m.vram.free_gb
    if needed > total:
        return "never", needed
    if needed <= free:
        return "fits_now", needed
    if needed <= free + m.evictable.get(where, 0.0):
        return "fits_after_eviction", needed
    return "wont_fit_now", needed


def _reasoning(recipe: Recipe) -> str | None:
    """ "on" | "off" | "template" for an LLM recipe, null otherwise.

    A recipe that thinks needs a much larger token budget than one that does
    not — below it, the answer is empty rather than short.
    """
    if not recipe.serves("llm"):
        return None
    if recipe.engine == "vllm":
        # No server-level switch: the chat template decides, and a caller
        # turns it off per request with chat_template_kwargs.
        return "template"
    from giq.adapters.llama_cpp import DEFAULT_REASONING, MODEL_REASONING

    return MODEL_REASONING.get(recipe.name, DEFAULT_REASONING)


def recipe_entry(recipe: Recipe, m: _Machine) -> dict[str, Any]:
    """One recipe, as ``GET /recipes`` reports it."""
    record = m.store.record_for(recipe.name)
    where = m.store.effective_device(recipe.name)
    gpu = resolve_device(where) if where else None
    fit, needed = _fit(recipe, where, m)
    inst = m.instances.get(recipe.name)
    avail, found = availability(recipe)
    return {
        "name": recipe.name,
        "label": recipe.display,
        "detail": recipe.detail,
        "modalities": list(recipe.modalities),
        "engine": recipe.engine,
        # The declared binary or interpreter that executes the engine.
        "runtime": runtime_of(recipe.engine),
        "aliases": list(recipe.aliases),
        "capabilities": list(recipe.capabilities),
        "vision": recipe.vision,
        "reasoning": _reasoning(recipe),
        "vram_gb": recipe.vram_gb,
        "measured": recipe.measured,
        "needed_gb": round(needed, 1),
        "lanes": recipe.lanes,
        "max_batch": recipe.max_batch,
        "voices": list(recipe.voices),
        "installed": installed(recipe.name),
        # ready | fetchable | manual | unfit (ADR-005), and what it is judged on.
        "availability": avail,
        "checks": [c.to_dict() for c in found],
        "weights": [weights_id(loc) for loc in locations(recipe.name)],
        "residency": {
            "policy": record.policy,
            "source": record.source,
            "reason": record.reason,
            # Kept loaded by default (the recipes' own residency, or
            # config.yaml's residents), whatever the operator has set since.
            "default_resident": recipe.name in m.defaults,
        },
        "card": {
            "device": record.device,
            "source": record.device_source,
            "effective": where,
            "index": gpu.index if gpu else None,
            "name": gpu.name if gpu else None,
        },
        "fit": fit,
        # When a job for it last completed, across every modality it serves.
        "last_used": m.last_used.get(recipe.name),
        "instance": {"id": inst.id, "state": inst.state, "residency": inst.residency}
        if inst
        else None,
    }


def card_budgets(m: _Machine) -> list[dict[str, Any]]:
    """The pinned set's budget on each card, in reload order.

    One per card: two cards' pinned recipes do not compete, and a single bar
    against one card's total is a sentence with two subjects.
    """
    pinned_by_device = m.store.pinned_by_device()
    default = selected_device()
    out = []
    for uuid, gpu in sorted(m.cards.items(), key=lambda kv: kv[1].index):
        pinned_gb = m.store.pinned_vram_gb(device=uuid)
        projected = pinned_gb + RESIDENT_SET_HEADROOM_GB + reserve_for(uuid)
        out.append(
            {
                "uuid": uuid,
                "index": gpu.index,
                "name": gpu.name,
                "total_gb": round(gpu.vram_total_gb, 2),
                "free_gb": round(gpu.vram_free_gb, 2),
                "reserve_gb": reserve_for(uuid),
                "pinned_gb": round(pinned_gb, 2),
                "pinned_needed_gb": round(projected, 2),
                "pinned_fits": projected <= gpu.vram_total_gb,
                "pinned": pinned_by_device.get(uuid, []),
                "default": bool(default and default.uuid == uuid),
            }
        )
    return out


async def _machine() -> _Machine:
    from giq.stats import get_stats

    cards = {gpu.uuid: gpu for gpu in await asyncio.to_thread(get_gpus)}
    last_used = dict(
        await get_stats().fetch(
            "SELECT recipe, MAX(ts) FROM jobs WHERE status='completed' GROUP BY recipe"
        )
    )
    return _Machine(cards, last_used)


@router.get("/recipes")
async def list_recipes() -> dict:
    """Every recipe, with its residency, card, fit, installation and instance.

    ``pinned`` is reload order — which recipe comes back first, and by the
    same token which is evicted last; ``cards`` is the pinned set's budget
    on each card.
    """
    m = await _machine()
    entries = await asyncio.to_thread(lambda: [recipe_entry(r, m) for r in all_recipes()])
    return {"recipes": entries, "cards": card_budgets(m), "pinned": m.store.residents()}


@router.get("/recipes/{name}")
async def get_recipe_entry(name: str) -> dict:
    recipe = _recipe_or_404(name)
    m = await _machine()
    return recipe_entry(recipe, m)


@router.get("/recipes/{name}/plan")
async def get_plan(name: str) -> dict:
    """What fetching recipe ``name`` takes, and whether it can run here
    (ADR-005 D3): this machine's checks, plus sizes, access and licence
    from the Hub."""
    from giq.plan import plan

    recipe = _recipe_or_404(name)
    return (await asyncio.to_thread(plan, recipe.name)).to_dict()


async def _after(recipe: Recipe, warnings: list[str]) -> dict:
    m = await _machine()
    return {"recipe": recipe_entry(recipe, m), "cards": card_budgets(m), "warnings": warnings}


def _card_label(uuid: str | None) -> str:
    """Human-readable card name for messages: "GPU 1 (RTX 5060 Ti)"."""
    gpu = resolve_device(uuid) if uuid else None
    if gpu is None:
        return "the default card"
    return f"GPU {gpu.index} ({gpu.name})"


@router.put("/recipes/{name}/residency")
async def set_residency(name: str, request: ResidencyRequest) -> dict:
    """Keep a recipe loaded (``pinned``), load it on demand (``auto``), or switch it ``off``.

    ``pinned`` keeps it loaded and brings it back on boot; ``auto`` loads it
    on demand and lets the scheduler evict it; ``off`` refuses jobs for it and
    blocks every load path until it is set back. Changes take effect on the
    next residents tick (a few seconds), which is also when a demoted recipe
    is actually unloaded — teardown waits out in-flight lane jobs first, so
    nothing is killed mid-generation.

    Pinning a set that cannot coexist on the card is refused with 409: the
    scheduler would thrash reloads forever trying to satisfy it. ``force``
    overrides that. Pinning a second LLM on one card is refused outright.
    """
    from giq.policy import PINNED
    from giq.stats import get_stats

    recipe = _recipe_or_404(name)
    store = get_policy_store()
    warnings: list[str] = []
    if request.policy == PINNED:
        target = store.effective_device(recipe.name)
        other_llm = store.resident_llm(exclude=recipe.name, device=target)
        if recipe.serves("llm") and other_llm is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{other_llm} is already pinned to the same card. One resident LLM per "
                    "card: unpin it, or bind one of them to another card."
                ),
            )
        fits, projected, total, device = store.pinned_fit(extra=recipe.name)
        where = _card_label(device)
        if not fits and not request.force:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"pinning {recipe.name} would need {projected:.1f}GB of {total:.1f}GB on "
                    f"{where} — the resident set there could never all load, and the scheduler "
                    "would retry forever. Unpin something, bind it to another card, or pass "
                    "force=true."
                ),
            )
        if not fits:
            warnings.append(
                f"pinned set on {where} needs {projected:.1f}GB of {total:.1f}GB — "
                "it cannot all load"
            )
    try:
        store.set(recipe.name, request.policy, request.reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    await get_stats().record_event("policy", f"{recipe.name} -> {request.policy}")
    return await _after(recipe, warnings)


@router.delete("/recipes/{name}/residency")
async def clear_residency(name: str) -> dict:
    """Back to the recipe's default residency. The card binding stays."""
    recipe = _recipe_or_404(name)
    get_policy_store().clear(recipe.name)
    return await _after(recipe, [])


@router.put("/recipes/{name}/card")
async def set_card(name: str, request: CardRequest) -> dict:
    """Bind a recipe to a card by index or UUID, or unbind it with ``null``.

    The binding decides where the recipe loads, which card's VRAM it is gated
    against, and which residents can be evicted to make room for it; it does
    not change residency. Stored as the UUID: an index is a position in this
    boot's enumeration and a binding has to outlive that. Refused (409) when
    the recipe could never fit that card — checked against the card's total,
    since a binding is durable — and when the card's pinned set would
    over-commit, unless ``force``.
    """
    from giq.stats import get_stats
    from giq.vram import can_load

    recipe = _recipe_or_404(name)
    store = get_policy_store()
    previous = store.device_for(recipe.name)
    device = None if request.device is None else str(request.device)
    try:
        store.set_device(recipe.name, device)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    warnings: list[str] = []
    target = store.effective_device(recipe.name)
    where = _card_label(target)
    fits_card, reason = await asyncio.to_thread(can_load, recipe.name, target)
    if not fits_card and "can never fit" in reason:
        store.set_device(recipe.name, previous)
        raise HTTPException(status_code=409, detail=f"{recipe.name} on {where}: {reason}")

    if store.policy_for(recipe.name) == "pinned":
        other_llm = store.resident_llm(exclude=recipe.name, device=target)
        if recipe.serves("llm") and other_llm is not None:
            store.set_device(recipe.name, previous)
            raise HTTPException(
                status_code=409,
                detail=f"{other_llm} is already pinned to {where}. One resident LLM per card.",
            )
        fits, projected, total, _dev = store.pinned_fit(extra=recipe.name)
        if not fits and not request.force:
            store.set_device(recipe.name, previous)
            raise HTTPException(
                status_code=409,
                detail=(
                    f"binding {recipe.name} to {where} would put {projected:.1f}GB of pinned "
                    f"recipes on a {total:.1f}GB card. Unpin something there, or pass force=true."
                ),
            )
        if not fits:
            warnings.append(
                f"pinned set on {where} needs {projected:.1f}GB of {total:.1f}GB — "
                "it cannot all load"
            )

    await get_stats().record_event("device", f"{recipe.name} -> {target or 'default'}")
    return await _after(recipe, warnings)
