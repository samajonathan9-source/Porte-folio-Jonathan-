"""Geometric ETH (collapse-threshold model) and the warp-shell dissociation.

The dissociation mechanism removes the decohering exterior layer (where the
local coherence C(r) falls below the ETH threshold) and re-measures the
topological persistence P_sig of the remaining coherent core. It models the
claim that a controlled collapse drives P_sig toward the universal kernel,
rather than to a singularity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..topology.rips import psig_from_points


class GeometricETH:
    """Collapse-threshold model per region (exterior decoheres fastest)."""

    def threshold(self, region: str, grad_curvature: float = 0.0) -> float:
        base = {"bulk": 0.15, "shell": 0.10, "exterior": 0.35}.get(region, 0.10)
        return base + 0.10 * grad_curvature


@dataclass(frozen=True)
class DissociationResult:
    P_sig_before: float
    P_sig_after: float
    n_before: int
    n_after: int
    removed_fraction: float
    scope: str = "controlled-collapse model on a warp-shell point cloud; not a physical collapse"


def _region_labels(regions: dict) -> list[str]:
    labels: list[str] = []
    for region in ("shell", "bulk", "exterior"):
        labels.extend([region] * int(regions.get(region, 0)))
    return labels


def dissociate_warp_shell(coords, regions, theta: float = 0.0, max_edge: float = 2.0,
                          eth: GeometricETH | None = None, coherence_profile=None) -> DissociationResult:
    """Remove the decohering layer and re-measure P_sig of the coherent core.

    coords: (n, 3) point cloud; regions: {'shell','bulk','exterior'} counts;
    coherence_profile: per-node local coherence C(r). Nodes whose C falls below
    the region ETH threshold are dissociated. theta/max_edge are carried for
    provenance with the engine contract (the measured quantity is P_sig).
    """
    eth = eth or GeometricETH()
    coords = np.asarray(coords, dtype=float)
    labels = _region_labels(regions)
    if coherence_profile is None:
        coherence_profile = np.ones(len(coords), dtype=float)
    coherence_profile = np.asarray(coherence_profile, dtype=float)

    before = psig_from_points(coords, max_edge=max_edge)
    keep = np.array(
        [coherence_profile[i] >= eth.threshold(labels[i]) for i in range(len(coords))],
        dtype=bool,
    )
    core = coords[keep]
    after = psig_from_points(core, max_edge=max_edge) if len(core) >= 4 else 0.0
    removed = 1.0 - (len(core) / max(1, len(coords)))
    return DissociationResult(
        P_sig_before=float(before),
        P_sig_after=float(after),
        n_before=int(len(coords)),
        n_after=int(len(core)),
        removed_fraction=float(removed),
    )
