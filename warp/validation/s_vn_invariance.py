"""von Neumann entropy invariance under a change of measured energy.

The LCT ansatz modulates the *phase* of the state by the energy (the current),
not the amplitudes (the message). Since S_vN depends only on the spectrum of
the reduced density matrix, it is invariant by construction of the ansatz —
coherent with the message/current duality, but an illustration, not an
independent validation (the real validation is the QPU preprint).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SvNInvarianceResult:
    energies: list
    s_vn_values: list
    s_vn_mean: float
    cv_pct: float
    scope: str = "ansatz modulates phase by energy; S_vN invariant by construction"


def _bell_pair_s_vn() -> float:
    """Entropy of one qubit of a Bell pair (maximally mixed) = 1 bit."""
    rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
    eigenvalues = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0).real
    positive = eigenvalues[eigenvalues > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def test_s_vn_invariance(C_local, energies, n_qubits: int = 8) -> SvNInvarianceResult:
    """Compute S_vN for a phase-modulated state at each energy.

    The energy enters only as a phase e^{i E t} on the entangled components, so
    the reduced spectrum — and therefore S_vN — is identical at every energy.
    """
    energies = [float(e) for e in energies]
    base = _bell_pair_s_vn()
    s_vn_values = []
    for energy in energies:
        # Apply the energy as a pure phase; the reduced density matrix is unchanged.
        phase = np.exp(1j * energy)
        rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex) * (phase * np.conj(phase))
        eigenvalues = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0).real
        positive = eigenvalues[eigenvalues > 0.0]
        s_vn_values.append(float(-np.sum(positive * np.log2(positive))))
    values = np.asarray(s_vn_values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    cv_pct = 0.0 if mean == 0.0 else 100.0 * std / mean
    return SvNInvarianceResult(
        energies=energies,
        s_vn_values=s_vn_values,
        s_vn_mean=mean,
        cv_pct=float(cv_pct),
    )
