# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""An image recipe's files, as sd-server takes them: its `weights.parts`,
resolved like any weights (giq.weights)."""

from __future__ import annotations

from dataclasses import dataclass

from giq.weights import path_of, recipe_of

IMAGE_PARTS = ("diffusion", "text_encoder", "vae", "lora")
_REQUIRED_IMAGE_PARTS = ("diffusion", "text_encoder", "vae")


@dataclass(frozen=True)
class ImageFiles:
    """An image recipe's files, as sd-server takes them."""

    diffusion: str
    text_encoder: str
    vae: str
    lora: str | None = None


def image_files(name: str) -> ImageFiles:
    """An image recipe's ``weights.parts``, paths resolved like any weights."""
    recipe = recipe_of(name)
    if recipe is None:
        raise ValueError(f"unknown recipe {name!r}: no recipe file defines it")
    files = {part: path_of(recipe.name, part) for part in IMAGE_PARTS}
    if missing := [p for p in _REQUIRED_IMAGE_PARTS if not files[p]]:
        raise ValueError(
            f"{recipe.name} names no {', '.join(f'weights.parts.{p}' for p in missing)} "
            "in its recipe file"
        )
    return ImageFiles(
        diffusion=str(files["diffusion"]),
        text_encoder=str(files["text_encoder"]),
        vae=str(files["vae"]),
        lora=files["lora"],
    )
