# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""From part colours to parts, on meshes small enough to reason about."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from giq_3d_child import parts  # noqa: E402


def strip(cells: int) -> tuple[np.ndarray, np.ndarray]:
    """A strip of `cells` squares along x, two triangles each, vertices welded."""
    xs = np.arange(cells + 1, dtype=float)
    vertices = np.concatenate(
        [np.stack([xs, 0 * xs, 0 * xs], 1), np.stack([xs, 1 + 0 * xs, 0 * xs], 1)]
    )
    faces = []
    for i in range(cells):
        a, b, c, d = i, i + 1, cells + 1 + i, cells + 2 + i
        faces += [(a, b, c), (b, d, c)]
    return vertices, np.array(faces)


def grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """An n by n sheet of squares, two triangles each, vertices welded."""
    ys, xs = np.mgrid[0 : n + 1, 0 : n + 1]
    vertices = np.stack([xs.reshape(-1), ys.reshape(-1), 0 * xs.reshape(-1)], 1).astype(float)
    faces = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            faces += [(a, a + 1, a + n + 1), (a + 1, a + n + 2, a + n + 1)]
    return vertices, np.array(faces)


def test_flat_colours_are_groups_and_the_blur_between_them_joins_the_nearer():
    red, blue = np.array([0.9, 0.1, 0.1]), np.array([0.1, 0.1, 0.9])
    colours = np.concatenate(
        [np.tile(red, (500, 1)), np.tile(blue, (300, 1)), [0.7 * red + 0.3 * blue]]
    )
    colours += np.random.default_rng(0).normal(0, 0.01, colours.shape)
    groups = parts.colour_groups(colours.clip(0, 1))
    assert np.bincount(groups).tolist() == [501, 300], "largest group first, the blur with red"


def test_faces_take_the_group_of_the_painted_points_nearest_them():
    vertices, faces = strip(10)
    centroids = vertices[faces].mean(axis=1)
    points = np.stack([np.linspace(0, 10, 200), np.full(200, 0.5), np.zeros(200)], 1)
    labels = parts.onto_faces(centroids, points, (points[:, 0] > 6).astype(int))
    assert labels.tolist() == [0] * 12 + [1] * 8


def test_cuts_laid_over_one_another_keep_every_cut():
    first = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    second = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    combined = parts.overlay([first, second])
    assert len(set(combined.tolist())) == 3
    assert combined[0] == combined[1] and combined[2] == combined[3] and combined[4] == combined[7]
    assert len({combined[0], combined[2], combined[4]}) == 3


def test_one_label_on_two_pieces_is_two_parts_and_a_speck_joins_its_surroundings():
    _, faces = strip(100)
    labels = np.zeros(200, int)
    labels[80:120] = 1  # a band in the middle parts the strip's two ends
    labels[30] = 1  # one face of the band's colour, far from the band
    tidied = parts.tidy(faces, labels)
    assert tidied.max() == 2, "left end, band, right end"
    assert tidied[30] == tidied[29] == tidied[31]
    assert len({tidied[0], tidied[100], tidied[199]}) == 3
    sizes = np.bincount(tidied)
    assert sizes.tolist() == sorted(sizes.tolist(), reverse=True), "largest first"


def test_no_more_parts_are_left_than_asked_for():
    _, faces = strip(100)
    labels = np.repeat(np.arange(10), 20)
    tidied = parts.tidy(faces, labels, most=4)
    assert tidied.max() + 1 == 4 and len(tidied) == 200


def test_an_asset_of_nothing_but_small_patches_still_comes_down_to_parts():
    _, faces = grid(60)
    labels = np.arange(len(faces)) // 72  # a hundred patches, each 1% of the faces
    tidied = parts.tidy(faces, labels, most=6)
    assert tidied.max() + 1 == 6 and np.bincount(tidied).min() > 72


def test_a_frayed_border_is_smoothed_to_the_grid():
    vertices, faces = strip(200)
    centroids = vertices[faces].mean(axis=1)
    labels = (centroids[:, 0] > 100).astype(int)
    labels[190:196] = 1  # a shard of the right part, just left of the border
    labels[204:208] = 0
    smoothed = parts.smooth(centroids, np.ones(len(faces)), labels, cells=100, reach=2)
    border = np.flatnonzero(np.diff(smoothed))
    assert len(border) == 1 and abs(border[0] - 199) <= 4


def test_parts_that_share_vertices_touch_and_say_where():
    vertices, faces = grid(20)
    labels = (vertices[faces].mean(axis=1)[:, 0] > 12).astype(int)
    (contact,) = parts.contacts(vertices, faces, labels)
    assert (contact["parts"], contact["points"]) == ([0, 1], 21)
    assert contact["centre"] == pytest.approx([12, 10, 0], abs=0.6)
    # A strip one square wide meets its other half at two vertices: not a contact.
    thin_v, thin_f = strip(40)
    assert parts.contacts(thin_v, thin_f, (np.arange(80) >= 40).astype(int)) == []
