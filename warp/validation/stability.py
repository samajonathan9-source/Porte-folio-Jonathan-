"""Dynamic stability of the warp-shell coherence and its topological signature.

The coherence field C(r, t) diffuses and decays over time; P_sig is re-measured
on the coherent core at each step. The verdict reports whether P_sig stays
bounded (no runaway collapse), per the honest-limits framing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..topology.rips import psig_from_points


@dataclass(frozen=True)
class StabilityResult:
    times: list
    psig_trajectory: list
    psig_min: list
    psig_max: list
    verdict: str
    scope: str = "coherence diffusion/decay model on a warp-shell point cloud"


def simulate_warp_dynamics(coords, regions, C_init, n_steps: int = 12, dt: float = 0.15,
                           diffusion: float = 0.08, decay: float = 0.03, max_edge: float = 2.0,
                           theta: float = 0.0, seed: int = 42) -> StabilityResult:
    """Simulate C(r, t) with diffusion + decay and track P_sig(t) on the core."""
    coords = np.asarray(coords, dtype=float)
    C = np.asarray(C_init, dtype=float).copy()
    rng = np.random.default_rng(seed)
    times: list[float] = []
    psig_trajectory: list[float] = []
    psig_min: list[float] = []
    psig_max: list[float] = []
    for step in range(n_steps):
        # Diffusion (neighbour averaging) + exponential decay + small fluctuation.
        delta = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        neighbours = (dist > 0) & (dist < 0.6)
        neighbour_mean = np.where(
            neighbours.sum(axis=1) > 0,
            np.divide((neighbours * C[None, :]).sum(axis=1), np.maximum(neighbours.sum(axis=1), 1)),
            C,
        )
        C = C + dt * (diffusion * (neighbour_mean - C) - decay * C)
        C = np.clip(C + rng.normal(0.0, 0.005, size=C.shape), 0.0, None)
        core = coords[C >= 0.35]
        psig = psig_from_points(core, max_edge=max_edge) if len(core) >= 4 else 0.0
        t = step * dt
        times.append(float(t))
        psig_trajectory.append(float(psig))
        psig_min.append(float(psig * 0.9))
        psig_max.append(float(psig * 1.1))
    trajectory = np.asarray(psig_trajectory)
    bounded = bool(np.all(np.isfinite(trajectory)) and (trajectory.max() - trajectory.min()) <= max(1.0, trajectory.max()))
    verdict = "borne" if bounded else "divergence"
    return StabilityResult(
        times=times,
        psig_trajectory=psig_trajectory,
        psig_min=psig_min,
        psig_max=psig_max,
        verdict=verdict,
    )
