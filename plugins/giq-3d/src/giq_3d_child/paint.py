# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Painting a mesh the client brings: its vertices and faces kept, a texture baked onto them.

The model paints in a frame of its own: the unit cube [-0.5, 0.5]^3, Z up.
The client's mesh is a GLB, Y up, at whatever scale and place it likes. So
the job carries a copy of the vertices into the model's frame, finds there
what colour each texel of the mesh's atlas stands for, and writes the
textures onto the client's own vertices, which never leave its frame. What
comes back has the vertices, faces and their order it went in with, so a
mesh that was checked, healed or rigged against stays the mesh it was.

Two frames, by where the paint's shape comes from:

  ``canonical``  the mesh is the shape (TRELLIS.2's texturing pipeline): it
                 is centred on its bounding box, scaled so its longest side
                 is just under 1, and turned Z up, exactly as upstream's
                 ``preprocess_mesh`` does before encoding it;
  ``generator``  the shape is a shape-only job's latent: the mesh is taken
                 to be in the frame that job's GLB was in, which is the
                 generator's cube turned Y up, and is only turned back.

UVs: a mesh that has them keeps them, and is painted through its own atlas,
which must lie in 0..1 and should not overlap (where two triangles share a
texel, one of them gets it). One without them, or asked to ``unwrap``, gets
a new atlas from xatlas, as the game export makes its own: glTF keeps one UV
per vertex, so every vertex on a seam is split in two, and the result says
for every vertex which input vertex it is a copy of (``vertex_map``).

The pieces that need the model come in as ``sample``; the rest is torch and
trimesh, so giq's suite runs it on the CPU with a stand-in for the model.
"""

import io
from dataclasses import dataclass

import numpy as np

MAX_FACES = 1_000_000
# UVs a hair outside the atlas are rounding, not tiling.
UV_SLACK = 1e-4


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float, the client's frame
    faces: np.ndarray  # (M, 3) int64
    uv: np.ndarray | None  # (N, 2), v up, or None
    normals: np.ndarray | None  # (N, 3) as the file gave them, or None


def sniff(data: bytes) -> str:
    """The mesh's file type: a GLB by its magic, else an OBJ by its records."""
    if data[:4] == b"glTF":
        return "glb"
    head = data[:65536].decode("utf-8", errors="replace")
    if any(line.lstrip().startswith(("v ", "f ")) for line in head.splitlines()):
        return "obj"
    raise ValueError("'mesh_b64' must be a GLB or an OBJ")


def read(data: bytes) -> Mesh:
    """The one mesh a GLB or OBJ holds, as stored. Raises ValueError naming what is wrong.

    A node transform in a GLB is applied, so the vertices are where the file
    places them; at the identity (what giq and trimesh write) they are the
    stored ones, bit for bit.
    """
    import trimesh

    kind = sniff(data)
    try:
        loaded = trimesh.load(io.BytesIO(data), file_type=kind, process=False, maintain_order=True)
    except Exception:
        raise ValueError(f"'mesh_b64' is not a {kind.upper()} giq can read") from None
    if isinstance(loaded, trimesh.Scene):
        meshes = [(n, g) for n, g in loaded.geometry.items() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("'mesh_b64' holds no triangles")
        if len(meshes) > 1:
            raise ValueError(
                f"'mesh_b64' holds {len(meshes)} meshes; a paint takes one: join them first"
            )
        name, mesh = meshes[0]
        nodes = loaded.graph.geometry_nodes.get(name, [])
        if len(nodes) != 1:
            raise ValueError(
                f"'mesh_b64' places its mesh {len(nodes)} times; a paint takes one placement"
            )
        # The geometry as stored, its normals included; moved only if its node moves it.
        transform, _ = loaded.graph.get(nodes[0])
        if not np.array_equal(transform, np.eye(4)):
            mesh = mesh.copy().apply_transform(transform)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("'mesh_b64' holds no triangles")
    if len(mesh.faces) > MAX_FACES:
        raise ValueError(
            f"'mesh_b64' has {len(mesh.faces)} faces; a paint takes {MAX_FACES} at most"
        )
    vertices = np.asarray(mesh.vertices)
    if not np.all(np.isfinite(vertices)):
        raise ValueError("'mesh_b64' has vertices that are not finite numbers")
    extent = vertices.max(0) - vertices.min(0)
    if float(extent.max()) <= 0:
        raise ValueError("'mesh_b64' has no extent: all its vertices are one point")
    uv = getattr(mesh.visual, "uv", None)
    uv = None if uv is None or len(uv) != len(vertices) else np.asarray(uv, dtype=np.float64)
    if uv is not None and (
        not np.all(np.isfinite(uv)) or uv.min() < -UV_SLACK or uv.max() > 1 + UV_SLACK
    ):
        raise ValueError(
            "'mesh_b64' has UVs outside 0..1, which a texture atlas cannot hold: "
            "send it without UVs, or with 'unwrap': true"
        )
    normals = mesh._cache.cache.get("vertex_normals")
    normals = None if normals is None or len(normals) != len(vertices) else np.asarray(normals)
    return Mesh(vertices, np.asarray(mesh.faces, dtype=np.int64), uv, normals)


def generator_axes(x: np.ndarray) -> np.ndarray:
    """Points of a GLB (Y up) in the generator's frame (Z up): the stock
    export's turn, undone."""
    x = np.asarray(x, dtype=np.float64).copy()
    x[:, 1], x[:, 2] = -x[:, 2].copy(), x[:, 1].copy()
    return x


def canonical(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """The centre and scale upstream's texturing pipeline puts a mesh in its cube with."""
    low, high = vertices.min(0), vertices.max(0)
    return (low + high) / 2, 0.99999 / float((high - low).max())


def to_model(vertices: np.ndarray, frame: str) -> np.ndarray:
    """A copy of the client's vertices in the model's frame; see the module docstring."""
    if frame == "canonical":
        centre, scale = canonical(vertices)
        return generator_axes((vertices - centre) * scale)
    if frame == "generator":
        return generator_axes(vertices)
    raise ValueError(f"unknown frame {frame!r}")


def paint(
    mesh: Mesh, frame: str, texture_size: int, sample, layout: dict, unwrap=False, device="cpu"
):
    """The mesh with a PBR texture baked onto it, and what was done to it.

    ``sample(P, Ns)`` is the model: for points [K, 3] on the mesh in the
    model's frame and their unit normals, the PBR attributes there [K, C]
    as ``layout`` lays them out, and the normals of a more detailed surface
    behind them [K, 3] to bake as a normal map, or None for no normal map.

    Returns (trimesh.Trimesh, info): ``info`` says whether the UVs were kept
    or made, and for made ones which input vertex each output vertex is.
    """
    import torch
    import trimesh
    from trimesh.visual import TextureVisuals

    from giq_3d_child import lowpoly

    model = to_model(mesh.vertices, frame)
    if mesh.uv is not None and not unwrap:
        vmap, faces, uv = None, mesh.faces, mesh.uv
    else:
        vmap, faces, uv = lowpoly.unwrap(mesh.vertices.astype(np.float32), mesh.faces, texture_size)
    pick = np.arange(len(model)) if vmap is None else np.asarray(vmap, dtype=np.int64)
    at = torch.from_numpy(pick).to(device)

    welded_v = torch.from_numpy(model).float().to(device)
    welded_f = torch.from_numpy(mesh.faces).long().to(device)
    # Smooth normals over the welded mesh, so a seam the atlas cuts shades as one.
    V, N = welded_v[at], lowpoly.smooth_normals(welded_v, welded_f)[at]
    UF = torch.from_numpy(np.asarray(faces, dtype=np.int64)).to(device)
    UV = torch.from_numpy(np.asarray(uv, dtype=np.float32)).to(device)

    mask, tri, P, Ns = lowpoly.texels(V, N, UF, UV, texture_size)
    attrs, behind = sample(P, Ns)
    normal = None if behind is None else lowpoly.tangent_normals(V, UV, UF, tri, Ns, behind)
    material = lowpoly.textures(attrs, layout, mask, texture_size, normal)

    normals = mesh.normals
    if normals is None:
        normals = lowpoly.smooth_normals(
            torch.from_numpy(mesh.vertices).double(), torch.from_numpy(mesh.faces).long()
        ).numpy()
    out = trimesh.Trimesh(
        vertices=mesh.vertices[pick],
        faces=np.asarray(faces),
        vertex_normals=normals[pick],
        visual=TextureVisuals(uv=np.asarray(uv), material=material),
        process=False,
    )
    info = {"uv": "kept" if vmap is None else "unwrapped"}
    if vmap is not None:
        info["vertex_map"] = [int(i) for i in vmap]
    return out, info
