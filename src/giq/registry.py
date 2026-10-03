# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The catalog: every recipe giq serves, by name (ADR-003).

Before the registry the model list lived in six places that each knew a part
of it, and they drifted: ``/capabilities`` advertised ``tts/kokoro-82m``, a
name absent from the VRAM table, so it fell through to a 22 GB default and
``wait_for_vram`` refused to load it at all on a 16 GB card. So: one record
per recipe, one place to add one, and everything else — VRAM gating, the
catalog, ``/capabilities``, the resident set, the dashboard — reads from
here. The records are the recipe files (``giq.recipes``), served as they are:
a recipe is what clients call, the scheduler keys on and the dashboard
shows, so nothing between the file and its readers reshapes it.

What config.yaml still says about recipes is residency — which are kept
loaded, in what order — and that is :func:`resident_defaults`'s, not the
recipe's.

Registration is not a whitelist. ``get_vram_requirement`` still falls back to
a default for an unknown name so an experimental model can be submitted
without a code change; it just won't be schedulable on a small card, which is
exactly the trap above. Give a model you intend to run a recipe file.
"""

from __future__ import annotations

import logging

from giq import recipes
from giq.recipes.schema import Recipe

logger = logging.getLogger(__name__)


def get_recipe(name: str) -> Recipe | None:
    """The recipe a client means by ``name`` (its name or an alias); None if unknown."""
    return recipes.current().get(str(name))


def all_recipes() -> list[Recipe]:
    """Every recipe, by name."""
    return sorted(recipes.current().recipes.values(), key=lambda r: r.name)


def recipes_serving(modality: str) -> list[Recipe]:
    """Every recipe that serves ``modality``, by name."""
    return recipes.current().serving(str(modality))


def resident_defaults() -> list[str]:
    """The default resident set, in reload-priority order.

    config.yaml's ``residents:`` replaces the recipes' own
    ``residency.priority`` entirely, in the order given. An entry may still
    be written ``worker/name`` as before ADR-003; the part before the slash
    is ignored.
    """
    from giq.config import get_config

    try:
        configured = getattr(get_config(), "residents", None)
    except Exception as e:  # config problems must not take the service down
        logger.warning(f"registry: config.yaml unreadable, using the recipes' residents: {e}")
        configured = None
    if configured:
        names = []
        for entry in configured:
            name = str(entry).rpartition("/")[2]
            recipe = get_recipe(name)
            if recipe is None:
                logger.warning(f"config residents: unknown recipe {entry!r}, ignoring")
                continue
            if recipe.name not in names:
                names.append(recipe.name)
        return names
    ranked = [r for r in all_recipes() if r.residency.priority is not None]
    ranked.sort(key=lambda r: r.residency.priority or 0)
    return [r.name for r in ranked]


def vram_for(name: str, *, default: float) -> float:
    """VRAM for recipe ``name``, or ``default`` when there is no such recipe.

    Adapters call this for their ``estimated_vram_gb``, which is what the
    eviction planner uses to size victims. Before the registry each worker
    kept its own table, so the planner and the VRAM gate could read different
    numbers for the same model — sd.cpp had zimage at 9GB while the gate
    required 13GB.
    """
    recipe = get_recipe(name)
    return recipe.vram_gb if recipe is not None else default


def lane_width_for(name: str, modality: str | None = None) -> int:
    """Concurrent jobs allowed on this recipe's resident lane."""
    recipe = get_recipe(name)
    if recipe is not None:
        return recipe.lanes
    from giq import plugins

    spec = plugins.modality(str(modality)) if modality is not None else None
    return spec.lane_width if spec is not None else 1


def reload_registry() -> list[Recipe]:
    """Re-read the recipe files; the catalog is whatever they now say."""
    recipes.reload()
    return all_recipes()
