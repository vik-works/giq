# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The game-ready export: a low-poly cage with the asset baked onto it.

The generator's mesh is millions of faces and has an inside: coil cores,
surfaces under plates, the inner wall of every shell. Measured on one asset,
simplifying it directly to 100k faces leaves 45% of them where nobody can
see, and at 1.5k the visible part falls apart. So the cage is not simplified
from that mesh. It is rebuilt from the outside:

  1. voxelise the surface and thicken it so no ray slips through;
  2. carve by line of sight: from 98 directions, march in from the border
     and mark every empty voxel a straight ray reaches. What no ray reaches
     is solid. Unlike a flood fill, this does not creep round corners into a
     cavity through a vent: only what can be seen is removed;
  3. close the small openings, so vent slots and gaps between coils become
     colour and normal map, as a modeller would paint them;
  4. take the solid's boundary with marching cubes: one closed skin with
     nothing behind it, where every face is one someone can see;
  5. simplify that skin, unwrap it, and bake onto it.

The bake finds, for every texel, the point of the full mesh it stands for,
by a ray shot inward from just outside the cage: the first thing it meets is
the outer surface, never the inside. There it reads the PBR attributes from
the generator's volume and the surface normal, written in the cage's tangent
space (+Y up, MikkTSpace-compatible: checked in Blender, where the map
lowers the shading error against the full mesh and a flipped green channel
raises it).

Measured on an RTX 5090 for one asset: 30 s for the skin, 30-45 s to
simplify and unwrap, 8 s and 1.6 GB of VRAM to bake.
"""

import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

from giq_3d_child import uvraster

# The voxel grid the skin is cut at, and how far the surface is thickened
# before carving; the thickening is given back afterwards.
RES, THICKEN = 384, 2
# How far outside the cage a bake ray starts, and how far inside it may land.
RAY_START, RAY_REACH = 0.02, 0.05
# Candidate voxels a carve direction steps at most; bounds its loop.
MAX_STEP = 2


def _dilate(x: torch.Tensor, r: int) -> torch.Tensor:
    return F.max_pool3d(x[None, None].float(), 2 * r + 1, stride=1, padding=r)[0, 0] > 0


def directions() -> list[tuple[int, int, int]]:
    """Every direction with components in -2..2, one per line through the origin."""
    span = range(-MAX_STEP, MAX_STEP + 1)
    return [
        (x, y, z)
        for x, y, z in itertools.product(span, span, span)
        if (x or y or z) and math.gcd(x, y, z) == 1
    ]


def carve(occupied: torch.Tensor, close: int) -> torch.Tensor:
    """The solid a surface voxelisation encloses, by line of sight.

    ``occupied`` is a cubic boolean grid with a free margin of at least
    THICKEN + close + 2 voxels on every side.
    """
    n = occupied.shape[0]
    free = ~_dilate(occupied, THICKEN)
    border = torch.zeros_like(occupied)
    edge = torch.tensor([0, n - 1], device=occupied.device)
    for axis in range(3):
        border.index_fill_(axis, edge, True)
    seen = torch.zeros_like(occupied)
    for d in directions():
        lit = border.clone()
        for _ in range(n // max(abs(c) for c in d) + 2):
            # The border is lit on every side, so what roll wraps round is lit already.
            lit |= torch.roll(lit, shifts=d, dims=(0, 1, 2)) & free
        seen |= lit
    # Give back the thickening, so the skin sits on the surface, not outside it.
    solid = ~_dilate(seen, THICKEN)
    if close:
        solid = ~_dilate(~_dilate(solid, close), close)
    return solid


def skin(vertices: torch.Tensor, close: int) -> tuple[np.ndarray, np.ndarray]:
    """One closed outward-facing surface round a mesh in [-0.5, 0.5]^3."""
    from scipy import ndimage
    from skimage import measure

    # Room for the thickening and the closing to grow into: a skin that
    # reached the edge of the array would be left open there.
    pad = THICKEN + close + 4
    n = RES + 2 * pad
    idx = ((vertices + 0.5) * RES).long().clamp(0, RES - 1) + pad
    occupied = torch.zeros((n, n, n), dtype=torch.bool, device=vertices.device)
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    solid = carve(occupied, close).cpu().numpy()
    # One object: crumbs the carve pinches off are not worth their faces.
    labels, count = ndimage.label(solid)
    if count > 1:
        sizes = ndimage.sum_labels(solid, labels, range(1, count + 1))
        solid = labels == (1 + int(np.argmax(sizes)))
    sdf = ndimage.distance_transform_edt(~solid) - ndimage.distance_transform_edt(solid)
    sdf = ndimage.gaussian_filter(sdf.astype(np.float32), 0.8)
    v, f, _, _ = measure.marching_cubes(sdf, level=0.0)
    v = ((v - pad + 0.5) / RES - 0.5).astype(np.float32)
    return v, outward(v, f)


def outward(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Faces wound so the surface's signed volume is positive.

    The bake shoots its rays from the outside, and a renderer culls what
    faces away, so a cage turned inside out bakes and draws its far side.
    """
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    if float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) < 0:
        f = f[:, ::-1]
    return np.ascontiguousarray(f)


def simplify(v: np.ndarray, f: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadric simplification in stages, pushing harder only if it stalls.

    In one step from half a million faces the simplifier stops well short
    of a few hundred; through 60k and 8k it arrives.
    """
    import pyfqmr

    for aggressiveness in (3, 5, 7):
        cv, cf = v, f
        for step in (60000, 8000, target):
            if len(cf) > step:
                simp = pyfqmr.Simplify()
                simp.setMesh(cv, cf)
                simp.simplify_mesh(
                    target_count=step,
                    aggressiveness=aggressiveness,
                    preserve_border=False,
                    max_iterations=300,
                    verbose=False,
                )
                cv, cf, _ = simp.getMesh()
        if len(cf) <= target * 1.1:
            break
    return np.asarray(cv, dtype=np.float32), outward(np.asarray(cv), np.asarray(cf))


def unwrap(v: np.ndarray, f: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A UV atlas for a cage: (vertex map, faces over it, uvs in 0..1)."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(v.astype(np.float32), f.astype(np.uint32))
    chart = xatlas.ChartOptions()
    # Few, large islands: a low-poly prop is hand-fixable only if its atlas is.
    chart.max_cost = 16.0
    chart.normal_deviation_weight = 0.75
    chart.normal_seam_weight = 1.0
    chart.roundness_weight = 0.01
    chart.max_iterations = 4
    pack = xatlas.PackOptions()
    pack.padding = max(2, size // 340)
    pack.resolution = size
    pack.bruteForce = True
    atlas.generate(chart_options=chart, pack_options=pack)
    vmap, faces, uvs = atlas[0]
    return vmap.astype(np.int64), faces.astype(np.int64), uvs.astype(np.float32)


def tangent_frame(tp, tuv, normal):
    """Per-sample tangent and bitangent of triangles [N, 3, 3] with uvs [N, 3, 2].

    Tangent along +u, bitangent along +v with v up, both made orthogonal to
    the shading normal: the frame a glTF normal map is read in.
    """
    dp1, dp2 = tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]
    du1, dv1 = (tuv[:, 1] - tuv[:, 0]).unbind(1)
    du2, dv2 = (tuv[:, 2] - tuv[:, 0]).unbind(1)
    r = du1 * dv2 - du2 * dv1
    r = torch.where(r.abs() < 1e-20, torch.full_like(r, 1e-20), r)
    tangent = (dp1 * dv2.unsqueeze(1) - dp2 * dv1.unsqueeze(1)) / r.unsqueeze(1)
    bitangent = (dp2 * du1.unsqueeze(1) - dp1 * du2.unsqueeze(1)) / r.unsqueeze(1)
    t = F.normalize(tangent - normal * (normal * tangent).sum(1, keepdim=True), dim=1)
    b = torch.linalg.cross(normal, t)
    # Mirrored UV islands run the other way round the normal.
    b = torch.where((b * bitangent).sum(1, keepdim=True) < 0, -b, b)
    return t, b


def on_triangle(corners: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Barycentric weights [N, 3] of points against triangles [N, 3, 3], clamped onto them."""
    e0, e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], point - corners[:, 0]
    d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
    d20, d21 = (e2 * e0).sum(1), (e2 * e1).sum(1)
    den = (d00 * d11 - d01 * d01).clamp(min=1e-20)
    b1 = ((d11 * d20 - d01 * d21) / den).clamp(0, 1)
    b2 = ((d00 * d21 - d01 * d20) / den).clamp(0, 1)
    w = torch.stack([(1 - b1 - b2).clamp(0, 1), b1, b2], dim=1)
    return w / w.sum(1, keepdim=True)


def export(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: dict,
    grid_size: int,
    aabb,
    target_faces: int,
    texture_size: int,
    close: int,
):
    """The generated asset as a low-poly trimesh with baked PBR and normal maps.

    The arguments up to ``aabb`` are what the stock export takes; the mesh
    comes back in the stock export's axes, so both land the same way up.
    """
    import cumesh
    import cv2
    import trimesh
    from flex_gemm.ops.grid_sample import grid_sample_3d
    from PIL import Image

    dev = vertices.device
    size = texture_size
    sv, sf = skin(vertices, close)
    cv_np, cf_np = simplify(sv, sf, target_faces)
    vmap, uf, uv = unwrap(cv_np, cf_np, size)
    cage_v = torch.from_numpy(cv_np).to(dev)
    cage_f = torch.from_numpy(cf_np.astype(np.int64)).to(dev)
    vmap_t = torch.from_numpy(vmap).to(dev)
    UF, UV = torch.from_numpy(uf).to(dev), torch.from_numpy(uv).to(dev)

    # Smooth normals on the welded cage, carried across the UV seams.
    fn = torch.linalg.cross(
        cage_v[cage_f[:, 1]] - cage_v[cage_f[:, 0]], cage_v[cage_f[:, 2]] - cage_v[cage_f[:, 0]]
    )
    vn = torch.zeros_like(cage_v).index_add_(0, cage_f.reshape(-1), fn.repeat_interleave(3, dim=0))
    V, N = cage_v[vmap_t], F.normalize(vn, dim=1)[vmap_t]

    # Which cage point each texel stands for.
    pos = torch.cat([UV * 2 - 1, torch.zeros_like(UV[:, :1]), torch.ones_like(UV[:, :1])], dim=1)
    rast, _ = uvraster.rasterize(None, pos.unsqueeze(0), UF, resolution=[size, size])
    mask = rast[0, ..., 3] > 0
    tri = rast[0, ..., 3][mask].long() - 1
    P = uvraster.interpolate(V.unsqueeze(0), rast, UF)[0][0][mask]
    Ns = F.normalize(uvraster.interpolate(N.unsqueeze(0), rast, UF)[0][0][mask], dim=1)

    # The point of the full mesh behind it.
    high = cumesh.CuMesh()
    high.init(vertices, faces)
    high.compute_vertex_normals()
    high_normals = high.read_vertex_normals()
    bvh = cumesh.cuBVH(vertices, faces)
    hit, face, depth = bvh.ray_trace(P + Ns * RAY_START, -Ns)
    found = (face >= 0) & (depth <= RAY_START + RAY_REACH)
    _, near_face, _ = bvh.unsigned_distance(P, return_uvw=True)
    face = torch.where(found, face, near_face.long())
    corner_ids = faces[face].long()
    corners = vertices[corner_ids]
    w = on_triangle(corners, torch.where(found.unsqueeze(1), hit, P))
    Q = (corners * w.unsqueeze(-1)).sum(1)
    high_normal = F.normalize((high_normals[corner_ids] * w.unsqueeze(-1)).sum(1), dim=1)
    high_normal = torch.where(
        (high_normal * Ns).sum(1, keepdim=True) < 0, -high_normal, high_normal
    )

    # PBR attributes from the generator's volume, as the stock export reads them.
    aabb = torch.as_tensor(aabb, dtype=torch.float32, device=dev)
    grid = torch.tensor([grid_size] * 3, dtype=torch.int32, device=dev)
    voxel = (aabb[1] - aabb[0]) / grid
    attrs = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid.tolist()]),
        grid=((Q - aabb[0]) / voxel).reshape(1, -1, 3),
        mode="trilinear",
    ).reshape(-1, attr_volume.shape[1])

    # The normal in the cage's tangent space. A normal turned past the
    # horizon of its texel cannot be stored; it is laid just above it.
    t, b = tangent_frame(V[UF[tri]], UV[UF[tri]], Ns)
    tangent_normal = F.normalize(
        torch.stack(
            [
                (high_normal * t).sum(1),
                (high_normal * b).sum(1),
                (high_normal * Ns).sum(1).clamp(min=0.05),
            ],
            dim=1,
        ),
        dim=1,
    )

    def image(values: torch.Tensor, fill: float) -> np.ndarray:
        """Texel values as an 8-bit image, top row first."""
        out = torch.full(
            (size, size, values.shape[1]), float(fill), dtype=torch.float32, device=dev
        )
        out[mask] = values
        return (out.flip(0).clamp(0, 1) * 255).round().byte().cpu().numpy()

    def part(name: str) -> torch.Tensor:
        return attrs[:, attr_layout[name]]

    # Fill the gutters between islands, so filtering at a seam finds colour.
    gutter = (~mask).flip(0).cpu().numpy().astype(np.uint8)
    base = cv2.inpaint(image(part("base_color"), 0), gutter, 3, cv2.INPAINT_TELEA)
    metal = cv2.inpaint(image(part("metallic"), 0), gutter, 1, cv2.INPAINT_TELEA)
    rough = cv2.inpaint(image(part("roughness"), 0), gutter, 1, cv2.INPAINT_TELEA)
    normal = cv2.inpaint(image(tangent_normal * 0.5 + 0.5, 0.5), gutter, 3, cv2.INPAINT_TELEA)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(base),
        metallicRoughnessTexture=Image.fromarray(
            np.stack([np.zeros_like(metal), rough, metal], axis=-1)
        ),
        normalTexture=Image.fromarray(normal),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        doubleSided=False,
    )

    def glb_axes(x: torch.Tensor) -> np.ndarray:
        # The stock export's axes: swap Y and Z, invert Y.
        x = x.cpu().numpy().copy()
        x[:, 1], x[:, 2] = x[:, 2].copy(), -x[:, 1].copy()
        return x

    return trimesh.Trimesh(
        vertices=glb_axes(V),
        faces=UF.cpu().numpy(),
        vertex_normals=glb_axes(N),
        visual=trimesh.visual.TextureVisuals(uv=UV.cpu().numpy(), material=material),
        process=False,
    )
