# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The UV rasteriser and the cage's carve, on the CPU against known answers.

Both are plain torch, so they run wherever torch is; an interpreter without
it (the engine's own carries it, a bare giq need not) skips them.
"""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from giq_3d_child import lowpoly, uvraster  # noqa: E402


def _bake(uvs, faces, attr, size):
    pos = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], -1)
    rast, _ = uvraster.rasterize(None, pos.unsqueeze(0), faces, resolution=[size, size])
    out, _ = uvraster.interpolate(attr.unsqueeze(0), rast, faces)
    return rast[0], out[0]


def _brute(uvs, faces, size):
    """Every triangle against every texel centre; the highest index wins."""
    ids = torch.zeros(size, size, dtype=torch.long)
    c = torch.arange(size, dtype=torch.float64) + 0.5
    sy, sx = torch.meshgrid(c, c, indexing="ij")
    p = uvs.double() * size
    for i, f in enumerate(faces.tolist()):
        (x0, y0), (x1, y1), (x2, y2) = p[f[0]], p[f[1]], p[f[2]]
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if d == 0:
            continue
        u = ((y1 - y2) * (sx - x2) + (x2 - x1) * (sy - y2)) / d
        v = ((y2 - y0) * (sx - x2) + (x0 - x2) * (sy - y2)) / d
        inside = (u >= -1e-9) & (v >= -1e-9) & (1 - u - v >= -1e-9)
        ids = torch.where(inside, torch.maximum(ids, torch.tensor(i + 1)), ids)
    return ids


@pytest.mark.parametrize("size", [7, 64, 257])
def test_a_quad_over_the_atlas_covers_every_texel_with_its_own_uv(size):
    uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    rast, out = _bake(uvs, faces, uvs, size)
    centre = (torch.arange(size) + 0.5) / size
    want = torch.stack(torch.meshgrid(centre, centre, indexing="xy"), -1)  # row is v, column is u
    assert bool((rast[..., 3] > 0).all())
    assert float((out - want).abs().max()) < 1e-6


@pytest.mark.parametrize(("cells", "size"), [(6, 96), (17, 160), (3, 200)])
def test_a_jittered_mesh_matches_brute_force(cells, size):
    torch.manual_seed(cells)
    g = torch.linspace(0.03, 0.97, cells + 1)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    uvs = (
        torch.stack([gx, gy], -1) + (torch.rand(cells + 1, cells + 1, 2) - 0.5) * 0.3 / cells
    ).reshape(-1, 2)
    idx = torch.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    faces = torch.cat(
        [torch.stack([a, b, c], -1).reshape(-1, 3), torch.stack([a, c, d], -1).reshape(-1, 3)]
    )
    faces = faces[torch.randperm(faces.shape[0])]
    rast, out = _bake(uvs, faces, uvs, size)
    got = rast[..., 3].long()
    assert int((got != _brute(uvs, faces, size)).sum()) == 0
    centre = (torch.arange(size) + 0.5) / size
    want = torch.stack(torch.meshgrid(centre, centre, indexing="xy"), -1)
    assert float((out - want)[got > 0].abs().max()) < 1e-5


def test_degenerate_and_oversized_triangles_do_not_break_it():
    uvs = torch.tensor([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [-0.5, -0.5], [1.5, -0.5], [0.5, 1.6]])
    rast, _ = _bake(uvs, torch.tensor([[0, 1, 2], [3, 4, 5]]), uvs, 32)
    assert set(rast[..., 3].unique().tolist()) == {0.0, 2.0}


def test_install_puts_the_rasteriser_where_nvdiffrast_is_imported_from(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "nvdiffrast", raising=False)
    monkeypatch.delitem(sys.modules, "nvdiffrast.torch", raising=False)
    uvraster.install()
    import nvdiffrast.torch as dr

    assert dr is uvraster


def _box(n, lo, hi, hole=None):
    """A hollow box as occupied voxels, optionally with a hole through its +x wall."""
    occ = torch.zeros((n, n, n), dtype=torch.bool)
    occ[lo:hi, lo:hi, lo:hi] = True
    occ[lo + 1 : hi - 1, lo + 1 : hi - 1, lo + 1 : hi - 1] = False
    if hole:
        mid = (lo + hi) // 2
        occ[hi - 1, mid - hole : mid + hole, mid - hole : mid + hole] = False
    return occ


def test_a_closed_shell_carves_to_a_filled_solid():
    solid = lowpoly.carve(_box(40, 12, 28), close=0)
    assert bool(solid[14:26, 14:26, 14:26].all()), "nothing sees the inside, so it is solid"
    assert not bool(solid[:8].any()) and not bool(solid[32:].any())


def test_only_what_a_straight_ray_reaches_is_carved_through_an_opening():
    solid = lowpoly.carve(_box(48, 12, 36, hole=3), close=0)
    mid = 24
    assert not bool(solid[30, mid, mid]), "straight in through the hole is seen"
    # Just inside the wall the hole is in, well to its side: part of the same
    # cavity, so a flood fill would reach it, but no direction is that shallow.
    assert bool(solid[31, 14, 14]), "beside the hole behind its wall is not seen"


def test_closing_seals_an_opening_narrower_than_twice_its_radius():
    assert bool(lowpoly.carve(_box(48, 12, 36, hole=2), close=3)[33, 24, 24])


def test_directions_are_one_per_line_through_the_origin():
    ds = lowpoly.directions()
    assert len(ds) == 98
    assert (2, 2, 0) not in ds and (1, 1, 0) in ds and (-1, -1, 0) in ds


def test_outward_turns_an_inside_out_surface():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    out = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert (lowpoly.outward(v, out) == out).all()
    assert (lowpoly.outward(v, out[:, ::-1]) == out).all()


def test_the_tangent_frame_follows_u_and_v_and_survives_a_mirrored_island():
    tri = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    n = torch.tensor([[0.0, 0, 1]])
    t, b = lowpoly.tangent_frame(tri, torch.tensor([[[0.0, 0], [1, 0], [0, 1]]]), n)
    assert torch.allclose(t, torch.tensor([[1.0, 0, 0]])) and torch.allclose(
        b, torch.tensor([[0.0, 1, 0]])
    )
    t, b = lowpoly.tangent_frame(tri, torch.tensor([[[1.0, 0], [0, 0], [1, 1]]]), n)
    assert torch.allclose(t, torch.tensor([[-1.0, 0, 0]])) and torch.allclose(
        b, torch.tensor([[0.0, 1, 0]])
    )


def test_a_point_off_a_triangle_is_clamped_onto_it():
    tri = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    w = lowpoly.on_triangle(tri, torch.tensor([[0.25, 0.25, 5.0]]))
    assert torch.allclose(w, torch.tensor([[0.5, 0.25, 0.25]]))
    w = lowpoly.on_triangle(tri, torch.tensor([[-3.0, -3.0, 0.0]]))
    assert float(w.min()) >= 0 and abs(float(w.sum()) - 1) < 1e-6
