# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Fitting views of an object to the camera rig Pixal3D's multi-view mode assumes.

The rig is the one the model ships as its example: four cameras a quarter
turn apart round the object, level with it, at one distance, 20 degrees of
view. Its first camera is the canonical front, and the mesh comes out in the
rig's frame: upright, squared to the axes, where a single picture gives a
mesh posed the way that picture was taken.

Views are used as given, never cropped or rescaled by the model, so each
has to be framed the way its camera says. Drawn or photographed views are
not: whoever made them filled the frame with whichever side they showed, so
a crate's long side and its short side come out the same width. Round a
level orbit the object's height is the one thing every view shares. Each
cut-out is therefore scaled to a common height and centred, which puts all
four at one distance.
"""

from __future__ import annotations

# The order views are given in, and the rig's camera for each.
VIEW_NAMES = ("front", "right", "back", "left")
# The side of a view in pixels, and the largest extent the object may take in
# it: the shipped example's objects span 840 to 940 of its 1024.
SIZE, FILL = 1024, 860
# Alpha above this is the object, for finding its box.
OPAQUE = 128


def object_box(alpha) -> tuple[int, int, int, int]:
    """The box of what is opaque in an alpha channel: left, top, right, bottom."""
    import numpy as np

    ys, xs = np.nonzero(np.asarray(alpha) > OPAQUE)
    if xs.size == 0:
        raise ValueError("a view has no object in it: its alpha channel is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_views(cutouts: list) -> list:
    """RGBA cut-outs of one object, each as a SIZE x SIZE view of the rig.

    Every object is brought to one height, then all are scaled together so
    the widest fills FILL pixels, and each is centred on a transparent frame.
    """
    from PIL import Image

    crops = [image.crop(object_box(image.getchannel("A"))) for image in cutouts]
    height = max(crop.height for crop in crops)
    widest = max(crop.width * height / crop.height for crop in crops)
    scale = FILL / max(height, widest)
    views = []
    for crop in crops:
        k = height / crop.height * scale
        size = (max(1, round(crop.width * k)), max(1, round(crop.height * k)))
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        view = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        view.paste(crop, ((SIZE - crop.width) // 2, (SIZE - crop.height) // 2), crop)
        views.append(view)
    return views
