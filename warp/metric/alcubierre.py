"""Canonical Alcubierre wall profile and the standard exotic-matter density.

The shape function f(r_s) is 1 inside the bubble and 0 outside, with a wall of
steepness eps around the radius R. The standard (Natario/Alcubierre) energy
density is negative on the wall — the exotic matter the LCT term aims to reduce.
"""
from __future__ import annotations

import numpy as np


def profile_tanh(r, R: float = 1.0, eps: float = 0.2):
    """Canonical Alcubierre wall shape: f(0) ~= 1, f(R) = 0.5, f -> 0 outside."""
    r = np.asarray(r, dtype=float)
    denominator = 2.0 * np.tanh(R / eps)
    numerator = np.tanh((r + R) / eps) - np.tanh((r - R) / eps)
    return numerator / denominator


def _profile_derivative(r, R: float, eps: float):
    r = np.asarray(r, dtype=float)
    denominator = 2.0 * np.tanh(R / eps)
    return (1.0 / (eps * denominator)) * (
        1.0 / np.cosh((r + R) / eps) ** 2 - 1.0 / np.cosh((r - R) / eps) ** 2
    )


def exotic_matter_baseline(profile, v: float = 1.0, R: float = 1.0, eps: float = 0.2,
                           n_samples: int = 1500, seed: int = 42) -> dict:
    """Estimate the mean negative energy density on the wall (standard result).

    rho_std ~ -(v^2 / 8 pi) (df/dr)^2 * (y^2+z^2)/r^2, evaluated by sampling the
    wall shell. The returned magnitude is a baseline to be reduced by Lambda_LCT.
    """
    rng = np.random.default_rng(seed)
    radius = rng.uniform(max(1e-6, R - 3 * eps), R + 3 * eps, size=n_samples)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    phi = np.arccos(rng.uniform(-1.0, 1.0, size=n_samples))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    def _f(rr):
        try:
            return np.asarray(profile(rr, R, eps), dtype=float)
        except TypeError:
            return np.asarray(profile(rr), dtype=float)

    f = _f(radius)
    # Finite-difference derivative of the supplied profile (agnostic to its form).
    h = 1e-4
    df = (_f(radius + h) - _f(radius - h)) / (2.0 * h)
    yz_over_r2 = (y * y + z * z) / np.maximum(radius * radius, 1e-12)
    rho = -(v * v / (8.0 * np.pi)) * (df * df) * yz_over_r2
    return {
        "mean_energy_density": float(np.mean(rho)),
        "min_energy_density": float(np.min(rho)),
        "integral_magnitude": float(np.sum(np.abs(rho))),
        "n_samples": int(n_samples),
        "f_wall_mean": float(np.mean(f)),
    }
