"""Universal topological kernel (preprint target P_sig ~= 1.80) and warp-shell geometry.

The warp shell is modelled as a sphere of coherent nodes (the wall), a flat
bulk and a decohering exterior. The universal kernel is the topological
signature the dissociation mechanism drives P_sig toward.
"""
from __future__ import annotations

import math

import numpy as np

UNIVERSAL_KERNEL_P_SIG = 1.80


def warp_shell_coords(R: float = 1.0, eps: float = 0.3, n_shell: int = 50,
                      n_bulk: int = 24, n_exterior: int = 12, seed: int = 42):
    """Build warp-shell coordinates and a region map (shell / bulk / exterior)."""
    rng = np.random.RandomState(seed)
    shell = []
    for _ in range(n_shell):
        th = rng.uniform(0, 2 * math.pi)
        ph = rng.uniform(0, math.pi)
        shell.append([R * math.sin(ph) * math.cos(th), R * math.sin(ph) * math.sin(th), R * math.cos(ph)])
    bulk = rng.uniform(-eps, eps, size=(n_bulk, 3)).tolist()
    exterior = rng.uniform(1.2, 2.0, size=(n_exterior, 3)).tolist()
    coords = np.array(shell + bulk + exterior, dtype=float)
    regions = {"shell": n_shell, "bulk": n_bulk, "exterior": n_exterior}
    return coords, regions


def psig_profile_universal(r, R: float = 1.0, width: float = 0.3,
                           amplitude: float = UNIVERSAL_KERNEL_P_SIG):
    """Target P_sig profile: a Gaussian kernel centred on the wall radius R.

    Used as the persistence target for the Lambda_LCT reduction and as the
    coherence-profile shape for the warp shell. Peaks at `amplitude` on the wall.
    """
    r = np.asarray(r, dtype=float)
    return amplitude * np.exp(-((r - R) ** 2) / (width ** 2))
