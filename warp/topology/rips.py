"""Small, deterministic Vietoris-Rips persistence for the warp lab.

Transparent GF(2) boundary reduction, deliberately favouring auditability over
large-scale performance. The engine uses finite H1 lifetime as its topological
signal (P_sig = longest finite H1 persistence).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def _ordered_simplices(distance: np.ndarray, max_edge: float) -> list[tuple[tuple[int, ...], float]]:
    n = distance.shape[0]
    simplices: list[tuple[tuple[int, ...], float]] = [((i,), 0.0) for i in range(n)]
    for i, j in combinations(range(n), 2):
        birth = float(distance[i, j])
        if birth <= max_edge:
            simplices.append(((i, j), birth))
    for i, j, k in combinations(range(n), 3):
        birth = float(max(distance[i, j], distance[i, k], distance[j, k]))
        if birth <= max_edge:
            simplices.append(((i, j, k), birth))
    return sorted(simplices, key=lambda item: (item[1], len(item[0]), item[0]))


def rips_persistence(distance: np.ndarray, max_edge: float | None = None) -> dict[str, Any]:
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    if not np.allclose(distance, distance.T, atol=1e-9):
        raise ValueError("distance must be symmetric")
    n = distance.shape[0]
    if n == 0:
        return {"diagrams": {"H0": [], "H1": []}, "betti": [0, 0, 0], "psig": 0.0, "n_finite_h1": 0}
    if max_edge is None:
        finite = distance[np.triu_indices(n, 1)]
        max_edge = float(np.max(finite)) if finite.size else 0.0
    simplices = _ordered_simplices(distance, float(max_edge))
    index = {simplex: idx for idx, (simplex, _) in enumerate(simplices)}
    births = [birth for _, birth in simplices]
    dims = [len(simplex) - 1 for simplex, _ in simplices]

    boundaries: list[set[int]] = []
    for simplex, _ in simplices:
        if len(simplex) == 1:
            boundaries.append(set())
            continue
        boundaries.append({index[tuple(v for pos, v in enumerate(simplex) if pos != removed)] for removed in range(len(simplex))})

    reduced_columns: dict[int, set[int]] = {}
    low_to_column: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    creators: set[int] = set()
    for col, raw_boundary in enumerate(boundaries):
        boundary = set(raw_boundary)
        while boundary and max(boundary) in low_to_column:
            boundary.symmetric_difference_update(reduced_columns[low_to_column[max(boundary)]])
        if not boundary:
            creators.add(col)
        else:
            low = max(boundary)
            reduced_columns[col] = boundary
            low_to_column[low] = col
            pairs.append((low, col))
            creators.discard(low)

    diagrams: dict[str, list[list[float | None]]] = {"H0": [], "H1": []}
    for birth_idx, death_idx in pairs:
        dimension = dims[birth_idx]
        if dimension in (0, 1):
            diagrams[f"H{dimension}"].append([float(births[birth_idx]), float(births[death_idx])])
    for creator in creators:
        dimension = dims[creator]
        if dimension in (0, 1):
            diagrams[f"H{dimension}"].append([float(births[creator]), None])

    tol = 1e-9
    finite_h1 = [death - birth for birth, death in diagrams["H1"] if death is not None and (death - birth) > tol]
    betti = [sum(1 for _, death in diagrams[f"H{dim}"] if death is None) for dim in (0, 1)] + [0]
    return {
        "diagrams": diagrams,
        "betti": betti,
        "psig": float(max(finite_h1, default=0.0)),
        "n_finite_h1": len(finite_h1),
        "max_edge": float(max_edge),
    }


def psig_from_points(points: np.ndarray, max_edge: float = 2.0) -> float:
    """Longest finite H1 persistence of a point cloud (the topological signal)."""
    points = np.asarray(points, dtype=float)
    delta = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, 0.0)
    return float(rips_persistence(distance, max_edge=max_edge)["psig"])
