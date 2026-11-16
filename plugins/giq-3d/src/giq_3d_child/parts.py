# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""From a model's part colours to the parts an asset is cut into.

A segmentation comes as colours on voxels: flat inside a part, blurred
where two meet, and different from one drawing to the next. This turns
any number of such drawings into one part number a face of the generated
mesh: colours to groups, groups onto faces, the drawings laid over one
another, borders smoothed, and the result tidied into parts that are one
connected piece each and none too small to be worth an export.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Colours closer than this (of 1, a channel on average) are one part.
SAME_COLOUR = 0.11
# A patch of faces smaller than this share of the asset is not a part of
# its own: it goes to whatever surrounds it.
SMALLEST_PART = 0.02


def colour_groups(colours: np.ndarray) -> np.ndarray:
    """A group number per colour: flat part colours, less the blur between them."""
    coarse = (colours * 31.999).astype(np.int64)
    _, inverse, counts = np.unique(
        coarse[:, 0] * 1024 + coarse[:, 1] * 32 + coarse[:, 2],
        return_inverse=True,
        return_counts=True,
    )
    means = np.stack([np.bincount(inverse, colours[:, c]) / counts for c in range(3)], axis=1)
    centres: list[np.ndarray] = []
    for index in np.argsort(-counts):
        if counts[index] < len(colours) * 0.005:
            break
        if all(np.abs(means[index] - c).mean() > SAME_COLOUR for c in centres):
            centres.append(means[index])
    distance = np.abs(colours[:, None, :] - np.array(centres)[None]).mean(axis=2)
    return distance.argmin(axis=1)


def onto_faces(centroids: np.ndarray, points: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each face takes the group most of the painted points nearest it have."""
    _, near = cKDTree(points).query(centroids, k=5, workers=-1)
    votes = groups[near]
    tally = np.stack([(votes == g).sum(axis=1) for g in range(groups.max() + 1)], axis=1)
    return tally.argmax(axis=1)


def overlay(labelings: list[np.ndarray]) -> np.ndarray:
    """Several segmentations of one mesh as one: faces are together only where all agree."""
    _, combined = np.unique(np.stack(labelings, axis=1), axis=0, return_inverse=True)
    return combined.reshape(-1)


def smooth(
    centroids: np.ndarray, area: np.ndarray, labels: np.ndarray, cells: int = 160, reach: int = 2
) -> np.ndarray:
    """Labels with the fray taken off their borders.

    A segmentation's border wanders by a few voxels of the model that drew
    it, and a part cut along it comes away with shards of its neighbour.
    Each face instead takes the label most of the surface within ``reach``
    cells of it has, on a grid of ``cells`` a side.
    """
    from scipy.ndimage import uniform_filter

    low, high = centroids.min(axis=0), centroids.max(axis=0)
    cell = np.clip(((centroids - low) / (high - low).max() * cells).astype(int), 0, cells - 1)
    flat = (cell[:, 0] * cells + cell[:, 1]) * cells + cell[:, 2]
    best = np.full(cells**3, -1.0, np.float32)
    winner = np.zeros(cells**3, np.int32)
    for label in range(labels.max() + 1):
        mine = labels == label
        share = np.bincount(flat[mine], area[mine], minlength=cells**3).astype(np.float32)
        share = uniform_filter(share.reshape(cells, cells, cells), 2 * reach + 1, mode="constant")
        better = share.reshape(-1) > best
        best[better], winner[better] = share.reshape(-1)[better], label
    return winner[flat]


def adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of faces sharing an edge, for a mesh whose vertices are welded."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    same = (edges[1:] == edges[:-1]).all(axis=1)
    return owner[:-1][same], owner[1:][same]


def _merged(borders, sizes: np.ndarray, label_of: np.ndarray, into: np.ndarray):
    """The pieces after each joins the piece ``into`` names for it.

    ``borders`` is the table of edges pieces share. Everything about the
    merged pieces follows from it and two small arrays, so the faces are
    not walked again: a mesh has millions of them and a few dozen pieces.
    """
    _, renumber = np.unique(into, return_inverse=True)
    n = int(renumber.max()) + 1
    table = borders.tocoo()
    row, col = renumber[table.row], renumber[table.col]
    apart = row != col
    table = coo_matrix((table.data[apart], (row[apart], col[apart])), shape=(n, n)).tocsr()
    # A merged piece keeps the label of its largest member.
    order = np.argsort(sizes)
    labels = np.zeros(n, label_of.dtype)
    labels[renumber[order]] = label_of[order]
    return table, np.bincount(renumber, sizes, minlength=n), labels, renumber


def tidy(faces: np.ndarray, labels: np.ndarray, most: int = 8) -> np.ndarray:
    """Labels as parts: one connected piece each, none too small, largest first.

    A label on two pieces that do not touch is two parts (a cab and a
    bucket the same colour). A patch too small to be a part joins what it
    borders most, and pieces of one label that then touch are one again: a
    part a speck had cut in two is whole. Crumbs that touch nothing go to
    the largest part, and if more than ``most`` are left the smallest is
    merged into its nearest neighbour until there are not.
    """
    a, b = adjacency(faces)
    same = labels[a] == labels[b]
    graph = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(len(faces),) * 2)
    count, piece = connected_components(graph, directed=False)
    sizes = np.bincount(piece, minlength=count).astype(np.float64)
    label_of = np.zeros(count, labels.dtype)
    label_of[piece] = labels
    cross = ~same
    borders = coo_matrix(
        (
            np.ones(2 * cross.sum()),
            (
                np.concatenate([piece[a[cross]], piece[b[cross]]]),
                np.concatenate([piece[b[cross]], piece[a[cross]]]),
            ),
        ),
        shape=(count, count),
    ).tocsr()
    del a, b, same, graph, cross
    floor = len(faces) * SMALLEST_PART

    def join(into: np.ndarray) -> None:
        nonlocal borders, sizes, label_of, piece
        borders, sizes, label_of, renumber = _merged(borders, sizes, label_of, into)
        piece = renumber[piece]

    for _ in range(32):
        small = sizes < floor
        # With nothing of full size there is nothing to join: the count
        # below is what brings such an asset down to parts.
        if not small.any() or small.all():
            break
        # Each small piece joins the piece of full size it shares most edges with.
        table = borders[:, np.flatnonzero(~small)]
        winner = np.flatnonzero(~small)[np.asarray(table.argmax(axis=1)).reshape(-1)]
        touched = np.asarray(table.sum(axis=1)).reshape(-1) > 0
        move = small & touched
        if not move.any():
            break
        into = np.arange(len(sizes))
        into[move] = winner[move]
        join(into)
        # Pieces of one label that now share a border were one part all along.
        table = borders.tocoo()
        alike = label_of[table.row] == label_of[table.col]
        graph = coo_matrix(
            (table.data[alike], (table.row[alike], table.col[alike])), shape=borders.shape
        )
        join(connected_components(graph, directed=False)[1])
    if (sizes >= floor).any():
        into = np.arange(len(sizes))
        into[sizes < floor] = sizes.argmax()
        join(into)
    while len(sizes) > most:
        least = int(sizes.argmin())
        row = np.asarray(borders[least].todense()).reshape(-1)
        into = np.arange(len(sizes))
        # Touching nothing, it goes to the largest part like any crumb.
        into[least] = int(row.argmax()) if row.any() else int(sizes.argmax())
        join(into)
    order = np.argsort(-sizes)
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    return rank[piece]


def contacts(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Where parts touch: for each pair that shares vertices, how many and their middle."""
    owners = [np.unique(faces[labels == part]) for part in range(int(labels.max()) + 1)]
    out = []
    for i in range(len(owners)):
        for j in range(i + 1, len(owners)):
            shared = np.intersect1d(owners[i], owners[j])
            if len(shared) >= 8:
                seam = vertices[shared]
                out.append({"parts": [i, j], "points": len(shared), "centre": seam.mean(axis=0)})
    return out
