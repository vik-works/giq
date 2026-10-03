# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Whether a recipe runs here, and which recipe a modality runs (ADR-005).

Every recipe has an availability, judged from what giq knows about this
machine without asking the network, cheap enough for every catalog poll:

- ``ready``: engine present, weights on disk, and it fits a card;
- ``fetchable``: the same, except the weights are not on disk and the
  recipe says where to fetch every missing file;
- ``manual``: weights missing, and not every missing file has a source;
- ``unfit``: it cannot run on this machine as it is (too large for every
  card, a card the engine cannot use, an engine binary that is missing).

The reasons come with it, as checks a person can act on. The full plan
before a fetch (sizes, access, licence) adds what only the Hub knows
(:mod:`giq.plan`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from giq import plugins
from giq.recipes.schema import Recipe

Status = Literal["ok", "warn", "fail"]
Availability = Literal["ready", "fetchable", "manual", "unfit"]


@dataclass(frozen=True)
class Check:
    """One thing a recipe needs, and whether this machine has it."""

    # engine | card | compute | weights | disk | access | licence | pinned | fetch
    check: str
    status: Status
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "status": self.status, "message": self.message}


def _engine(recipe: Recipe) -> Check:
    from giq.engines import all_engines

    engine = plugins.engine(recipe.engine)
    spec = all_engines().get(recipe.engine)
    if engine is None or engine.binary is None or spec is None:
        return Check("engine", "ok", f"{recipe.engine} runs in giq's own interpreter")
    if not os.path.exists(spec.binary):
        where = f" or set {engine.binary.env}" if engine.binary.env else ""
        return Check(
            "engine",
            "fail",
            f"the {recipe.engine} binary is not at {spec.binary}: install it, and declare "
            f"its path under `engines:` in config.yaml{where}",
        )
    return Check("engine", "ok", f"{recipe.engine} at {spec.binary}")


def _cards(recipe: Recipe) -> list[Check]:
    """Is there a card large enough, and one the engine can use?"""
    from giq.gpus import compute_capabilities, get_gpus
    from giq.vram import margin_of, reserve_for

    cards = get_gpus()
    if not cards:
        return [Check("card", "warn", "no GPU seen; whether it fits is unknown")]
    engine = plugins.engine(recipe.engine)
    checks = []
    if engine is not None and engine.check is not None:
        capabilities = compute_capabilities()
        reasons = [
            engine.check(recipe, cap)
            for card in cards
            if (cap := capabilities.get(card.uuid.lower())) is not None
        ]
        usable = [r for r in reasons if r is None]
        if reasons and not usable:
            checks.append(Check("compute", "fail", str(reasons[0])))
            cards = []
        elif len(usable) < len(reasons):
            checks.append(Check("compute", "warn", "some cards cannot run it"))
    needed = recipe.vram_gb + margin_of(recipe.name, recipe.vram_gb)
    room = max((c.vram_total_gb - reserve_for(c.uuid) for c in cards), default=0.0)
    if cards and needed > room:
        checks.append(
            Check(
                "card",
                "fail",
                f"needs {needed:.1f} GB with its margin; the largest card has {room:.1f} GB",
            )
        )
    elif cards:
        checks.append(Check("card", "ok", f"needs {needed:.1f} GB of {room:.1f} GB"))
    return checks


def _weights(recipe: Recipe) -> Check:
    from giq.storage import missing
    from giq.weights import locations

    if not locations(recipe.name):
        return Check("weights", "fail", "the recipe names no weights")
    gone = missing(recipe.name)
    if not gone:
        return Check("weights", "ok", "on disk")
    sources = _sources(recipe)
    unsourced = [loc for loc in gone if not sources.get(loc.part)]
    if unsourced:
        what = ", ".join(loc.path or loc.repo or "?" for loc in unsourced)
        return Check(
            "weights",
            "fail",
            f"not on disk, and the recipe says nowhere to fetch {what} from: place it there",
        )
    return Check("weights", "warn", "not on disk; giq can fetch them")


def _sources(recipe: Recipe) -> dict[str | None, str | None]:
    """Part (None for the main weights) -> its source."""
    if recipe.weights is None:
        return {}
    return {
        None: recipe.weights.source,
        **{part: p.source for part, p in recipe.weights.parts.items()},
    }


def checks(recipe: Recipe) -> list[Check]:
    """What this machine says about ``recipe``, without the network."""
    return [_engine(recipe), *_cards(recipe), _weights(recipe)]


def availability_of(found: list[Check]) -> Availability:
    weights = next(c for c in found if c.check == "weights")
    if any(c.status == "fail" for c in found if c.check != "weights"):
        return "unfit"
    if weights.status == "ok":
        return "ready"
    return "fetchable" if weights.status == "warn" else "manual"


def availability(recipe: Recipe) -> tuple[Availability, list[Check]]:
    found = checks(recipe)
    return availability_of(found), found


def ready(modality: str) -> list[str]:
    """The recipes a request for ``modality`` can run here, preferred first.

    Kept warm first, in the resident set's order, then the on-demand ones
    by name: an operator makes a recipe the default by keeping it warm.
    Switched-off recipes refuse jobs, so they are not offered.
    """
    from giq.policy import OFF, get_policy_store
    from giq.registry import recipes_serving

    store = get_policy_store()
    serving = {
        r.name: r
        for r in recipes_serving(modality)
        if store.policy_for(r.name) != OFF and availability(r)[0] == "ready"
    }
    warm = [n for n in store.residents() if n in serving]
    return warm + sorted(n for n in serving if n not in warm)


def default_recipe(modality: str) -> str | None:
    """The recipe a request for ``modality`` that names none runs on."""
    found = ready(modality)
    return found[0] if found else None
