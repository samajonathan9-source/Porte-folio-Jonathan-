"""Coherence-profile shapes for the warp shell."""
from __future__ import annotations

import numpy as np


def gaussian_shell_profile(r, params):
    """Gaussian coherence profile centred on the wall.

    params = [amplitude, centre, width]. Returns a coherence value in [0, +inf)
    evaluated at each radius r; the dissociation uses it as the local coherence
    C(r) that the wall shape modulates (an honest modelling assumption).
    """
    amplitude, centre, width = float(params[0]), float(params[1]), float(params[2])
    r = np.asarray(r, dtype=float)
    return amplitude * np.exp(-((r - centre) ** 2) / (width ** 2))
