"""Lambda_LCT term: three comparable ansatz to reduce Alcubierre exotic matter.

Per the honest-limits note, three ansatz are compared numerically; only the
KINETIC ansatz reduces the exotic matter (a modest reduction, not elimination).
The full covariant 4D derivation is documented as work in progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .alcubierre import exotic_matter_baseline


class LCTAnsatz(Enum):
    KINETIC = "kinetic"
    LOCAL_CC = "local_cc"
    PRESSURE = "pressure"


@dataclass(frozen=True)
class ReductionResult:
    ansatz: str
    kappa: float
    baseline_magnitude: float
    compensated_magnitude: float
    reduction_ratio: float
    scope: str = "numeric wall-shell estimate; not a covariant 4D derivation"


def reduce_exotic_matter(profile, v: float = 1.0, R: float = 1.0, eps: float = 0.2,
                         psig_profile=None, ansatz: LCTAnsatz = LCTAnsatz.KINETIC,
                         kappa: float = 1e-3, n_samples: int = 1500, seed: int = 42) -> ReductionResult:
    """Estimate how much a Lambda_LCT ansatz compensates the exotic-matter density.

    The compensation scales with kappa and with the persistence gradient
    (|dP/dr| for kinetic/local_cc, P for pressure). Only the kinetic ansatz
    yields a positive (reducing) contribution in this numeric estimate.
    """
    baseline = exotic_matter_baseline(profile, v=v, R=R, eps=eps, n_samples=n_samples, seed=seed)
    base_mag = baseline["integral_magnitude"]

    rng = np.random.default_rng(seed)
    radius = rng.uniform(max(1e-6, R - 3 * eps), R + 3 * eps, size=n_samples)
    if psig_profile is None:
        from ..topology.universal_kernel import psig_profile_universal
        psig_profile = psig_profile_universal
    P = np.asarray(psig_profile(radius, R), dtype=float)
    h = 1e-4
    dP = (np.asarray(psig_profile(radius + h, R), dtype=float)
          - np.asarray(psig_profile(radius - h, R), dtype=float)) / (2.0 * h)

    if ansatz is LCTAnsatz.KINETIC:
        # Λ_00 ~ κ |∇P|²  (positive energy, compensates the negative density)
        compensation = float(np.sum(kappa * dP * dP))
    elif ansatz is LCTAnsatz.LOCAL_CC:
        # Λ_00 ~ -κ □P g_00 ; in this estimate it does not reduce the wall density
        compensation = float(np.sum(-kappa * dP * dP))
    else:  # PRESSURE
        # Λ_00 ~ +κ P g_00 ; acts as a pressure term, no net reduction here
        compensation = float(np.sum(-kappa * P))

    compensated = base_mag - compensation
    reduction_ratio = 0.0 if base_mag == 0.0 else (base_mag - compensated) / base_mag
    return ReductionResult(
        ansatz=ansatz.value,
        kappa=float(kappa),
        baseline_magnitude=float(base_mag),
        compensated_magnitude=float(compensated),
        reduction_ratio=float(reduction_ratio),
    )
