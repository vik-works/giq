# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The shape a shape-only job hands its client, to be painted later.

TRELLIS.2 makes an asset in two halves: the shape latent (a sparse set of
voxels on a 64 grid, 32 numbers each, which the shape decoder turns into the
surface) and the texture latent sampled on top of it. A shape-only job stops
after the first and returns it with the mesh; a paint job given it back
samples the texture on that very latent, so the paint belongs to the shape
the client judged, not to a re-encoding of its mesh.

The artefact travels with the client, not in a store on the server: giq
keeps no payloads between jobs, and an on-demand instance is evicted
whenever another recipe needs the card, which is exactly what happens while
a client checks and heals a shape between the two jobs. It does not expire
by time. It is refused when it was made by another recipe, another shape
decoder or another version of this format, which is when it would no longer
decode to the shape it was made from.

Layout: ``MAGIC``, a little-endian u32 header length, the header as JSON,
then the zlib-compressed payload: the voxel coordinates as u16 (tokens x 3)
followed by the latent as f32 (tokens x channels). The header carries a
SHA-256 of the payload, so a truncated or altered artefact is refused rather
than decoded into something else. A voxel is 134 bytes before compression:
an object of 40,000 surface voxels on the 64 grid is about 5.4 MB, and
base64 adds a third.

Pure numpy and the standard library, so giq's suite tests it without the engine.
"""

import base64
import binascii
import hashlib
import json
import struct
import zlib

import numpy as np

MAGIC = b"GIQSHAPE"
VERSION = 1
# A bound on a latent's voxels, against a header that claims nonsense. The
# generator's cascade holds the 1536 stage under 49,152 but lets the 1024
# stage run past it; a surface on the 64 grid is tens of thousands.
MAX_TOKENS = 200_000
CHANNELS = 32
MAX_BYTES = MAX_TOKENS * (6 + 4 * CHANNELS)


class StaleShape(ValueError):
    """An artefact that is intact but no longer decodes to its shape here."""


def pack(
    coords: np.ndarray,
    feats: np.ndarray,
    *,
    recipe: str,
    decoder: str,
    resolution: int,
    seed: int,
) -> bytes:
    """A shape latent as bytes. ``coords`` is (tokens, 3) on the latent's
    grid, without the batch column; ``feats`` (tokens, channels)."""
    coords = np.ascontiguousarray(coords, dtype=np.uint16)
    feats = np.ascontiguousarray(feats, dtype="<f4")
    if coords.ndim != 2 or coords.shape[1] != 3 or feats.shape[0] != coords.shape[0]:
        raise ValueError("a shape latent is (tokens, 3) coordinates and (tokens, channels) values")
    raw = coords.astype("<u2").tobytes() + feats.tobytes()
    header = {
        "version": VERSION,
        "recipe": recipe,
        "decoder": decoder,
        "resolution": int(resolution),
        "seed": int(seed),
        "tokens": int(coords.shape[0]),
        "channels": int(feats.shape[1]),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    head = json.dumps(header, sort_keys=True).encode()
    return MAGIC + struct.pack("<I", len(head)) + head + zlib.compress(raw, 6)


def encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode(value) -> bytes:
    """``shape_b64`` from a task, as bytes."""
    if not isinstance(value, str) or not value:
        raise ValueError("'shape_b64' must be the shape_b64 a shape-only job returned")
    try:
        data = base64.b64decode(value.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("'shape_b64' is not base64") from None
    if len(data) > MAX_BYTES:
        raise ValueError(f"'shape_b64' is larger than {MAX_BYTES // 2**20} MB")
    return data


def header(data: bytes) -> dict:
    """The header of an artefact, checked for form; the payload is not read."""
    if not data.startswith(MAGIC) or len(data) < len(MAGIC) + 4:
        raise ValueError("'shape_b64' is not a shape a shape-only job returned")
    (length,) = struct.unpack_from("<I", data, len(MAGIC))
    start = len(MAGIC) + 4
    try:
        head = json.loads(data[start : start + length])
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("'shape_b64' has a damaged header") from None
    if not isinstance(head, dict):
        raise ValueError("'shape_b64' has a damaged header")
    return head


def unpack(data: bytes, *, recipe: str, decoder: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """(header, coords, feats) of an artefact made by this recipe and decoder.

    Raises StaleShape for one made elsewhere or by another format, and
    ValueError for one that is damaged."""
    head = header(data)
    if head.get("version") != VERSION:
        raise StaleShape(
            f"'shape_b64' is format {head.get('version')!r}, this giq reads {VERSION}: "
            "make the shape again"
        )
    if head.get("recipe") != recipe:
        raise StaleShape(
            f"'shape_b64' was made by {head.get('recipe')!r}, not {recipe!r}: "
            "paint it with the recipe that made it, or without shape_b64"
        )
    if head.get("decoder") != decoder:
        raise StaleShape(
            "'shape_b64' was made with another shape decoder than this recipe now loads: "
            "make the shape again, or paint without shape_b64"
        )
    tokens, channels = head.get("tokens"), head.get("channels")
    if not (isinstance(tokens, int) and 0 < tokens <= MAX_TOKENS and channels == CHANNELS):
        raise ValueError("'shape_b64' has a damaged header")
    start = len(MAGIC) + 4 + struct.unpack_from("<I", data, len(MAGIC))[0]
    try:
        raw = zlib.decompress(data[start:], bufsize=tokens * (6 + 4 * channels))
    except zlib.error:
        raise ValueError("'shape_b64' is damaged") from None
    if len(raw) != tokens * (6 + 4 * channels) or hashlib.sha256(raw).hexdigest() != head.get(
        "sha256"
    ):
        raise ValueError("'shape_b64' is damaged")
    coords = np.frombuffer(raw, dtype="<u2", count=tokens * 3).reshape(tokens, 3)
    feats = np.frombuffer(raw, dtype="<f4", offset=tokens * 6).reshape(tokens, channels)
    return head, coords.astype(np.int32), feats.astype(np.float32)
