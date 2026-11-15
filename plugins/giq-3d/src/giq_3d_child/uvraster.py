# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""A UV-space rasteriser in plain PyTorch, standing in for nvdiffrast.

o_voxel's GLB export bakes its texture by rasterising the mesh in UV space:
for every texel, which triangle covers it and where on that triangle. That
is the only thing the export asks of nvdiffrast, whose licence allows
research and evaluation use only. Charts in a UV atlas do not overlap, so
none of what makes a rasteriser hard is needed here: no depth test, no
perspective, no antialiasing, no gradients.

The three names o_voxel uses are provided with nvdiffrast's conventions, so
`install()` can put this module in its place:

    rast[..., 0:2]  barycentrics (u, v): attr = u*a0 + v*a1 + (1-u-v)*a2
    rast[..., 3]    triangle index + 1, 0 where nothing was drawn
    row 0 of the image is y = -1 in clip space
"""

import sys
import types

import torch

# Candidate texels evaluated per step; bounds the working memory.
_BUDGET = 1 << 23


class RasterizeCudaContext:
    def __init__(self, device=None):
        self.device = device


def _pixel_xy(pos: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Clip-space vertices [V, 4] as continuous pixel coordinates [V, 2]."""
    xy = pos[:, :2] / pos[:, 3:4]
    scale = torch.tensor([width, height], device=pos.device, dtype=torch.float64)
    return (xy.double() + 1.0) * 0.5 * scale


def _barycentrics(tri_xy: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor):
    """(u, v, w, d) of sample points against triangles [..., 3, 2]; d = 2 * area."""
    x0, y0 = tri_xy[..., 0, 0], tri_xy[..., 0, 1]
    x1, y1 = tri_xy[..., 1, 0], tri_xy[..., 1, 1]
    x2, y2 = tri_xy[..., 2, 0], tri_xy[..., 2, 1]
    d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    safe = torch.where(d == 0, torch.ones_like(d), d)
    u = ((y1 - y2) * (sx - x2) + (x2 - x1) * (sy - y2)) / safe
    v = ((y2 - y0) * (sx - x2) + (x0 - x2) * (sy - y2)) / safe
    return u, v, 1.0 - u - v, d


def rasterize(ctx, pos: torch.Tensor, tri: torch.Tensor, resolution, **_):
    """Which triangle covers each texel centre, and where: [1, H, W, 4].

    Two passes. The first writes triangle ids with a maximum, so a texel on
    an edge two triangles share gets one of them deterministically. The
    second computes the barycentrics from the id that won, which keeps the
    two halves of each texel's answer from ever coming from different
    triangles.
    """
    height, width = int(resolution[0]), int(resolution[1])
    device = pos.device
    tri = tri.long()
    xy = _pixel_xy(pos[0], height, width)  # [V, 2]
    corners = xy[tri]  # [F, 3, 2]

    # Texel k has its centre at k + 0.5; the ones a triangle's box can hold.
    upper = torch.tensor([width - 1, height - 1], device=device)
    lo = torch.ceil(corners.amin(dim=1) - 0.5).long().clamp(min=0)
    hi = torch.minimum(torch.floor(corners.amax(dim=1) - 0.5).long(), upper)
    extent = (hi - lo + 1).clamp(min=0)
    side = extent.amax(dim=1)
    side = torch.where((extent > 0).all(dim=1), side, torch.zeros_like(side))

    ids = torch.zeros(height * width, dtype=torch.int64, device=device)
    # Slightly generous at the edges: a texel centre exactly on a shared edge
    # must belong to someone, and belonging to both is settled by the maximum.
    eps = 1e-9

    bound = 1
    while True:
        prev = bound // 2 if bound > 1 else 0
        chosen = torch.nonzero((side > prev) & (side <= bound)).squeeze(1)
        if chosen.numel():
            oy, ox = torch.meshgrid(
                torch.arange(bound, device=device),
                torch.arange(bound, device=device),
                indexing="ij",
            )
            ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
            step = max(1, _BUDGET // (bound * bound))
            for start in range(0, chosen.numel(), step):
                f = chosen[start : start + step]
                cx = lo[f, 0:1] + ox
                cy = lo[f, 1:2] + oy
                in_box = (cx <= hi[f, 0:1]) & (cy <= hi[f, 1:2])
                u, v, w, d = _barycentrics(
                    corners[f].unsqueeze(1), cx.double() + 0.5, cy.double() + 0.5
                )
                inside = in_box & (d != 0) & (u >= -eps) & (v >= -eps) & (w >= -eps)
                target = (cy * width + cx)[inside]
                value = (f.unsqueeze(1) + 1).expand_as(cx)[inside]
                ids.scatter_reduce_(0, target, value, reduce="amax")
        if bound >= max(height, width):
            break
        bound *= 2

    rast = torch.zeros(height * width, 4, dtype=torch.float32, device=device)
    covered = torch.nonzero(ids > 0).squeeze(1)
    for start in range(0, covered.numel(), _BUDGET):
        texel = covered[start : start + _BUDGET]
        f = ids[texel] - 1
        sx = (texel % width).double() + 0.5
        sy = torch.div(texel, width, rounding_mode="floor").double() + 0.5
        u, v, _w, _d = _barycentrics(corners[f], sx, sy)
        rast[texel, 0] = u.float()
        rast[texel, 1] = v.float()
        rast[texel, 3] = (f + 1).float()
    return rast.view(1, height, width, 4), None


def interpolate(attr: torch.Tensor, rast: torch.Tensor, tri: torch.Tensor, **_):
    """Vertex attributes [1, V, C] at each drawn texel: [1, H, W, C]; zero elsewhere."""
    tri = tri.long()
    _, height, width, _ = rast.shape
    flat = rast.view(-1, 4)
    out = torch.zeros(height * width, attr.shape[-1], dtype=attr.dtype, device=attr.device)
    covered = torch.nonzero(flat[:, 3] > 0).squeeze(1)
    for start in range(0, covered.numel(), _BUDGET):
        texel = covered[start : start + _BUDGET]
        corners = attr[0][tri[flat[texel, 3].long() - 1]]  # [N, 3, C]
        u = flat[texel, 0:1].to(attr.dtype)
        v = flat[texel, 1:2].to(attr.dtype)
        out[texel] = u * corners[:, 0] + v * corners[:, 1] + (1.0 - u - v) * corners[:, 2]
    return out.view(1, height, width, -1), None


def install() -> None:
    """Make `import nvdiffrast.torch as dr` resolve to this module.

    Call before o_voxel is imported. A real nvdiffrast, if one is installed,
    is then never loaded.
    """
    this = sys.modules[__name__]
    package = types.ModuleType("nvdiffrast")
    setattr(package, "torch", this)  # noqa: B010 — a module has no such attribute to assign
    sys.modules["nvdiffrast"] = package
    sys.modules["nvdiffrast.torch"] = this
