"""Spanwise inflow computation using Prandtl tip-loss and table-lookup iteration."""
from __future__ import annotations

import numpy as np

from .airfoil import AirfoilTable
from .config import RotorConfig
from .lookup import lookup_coefficients


def compute_inflow(
    r: np.ndarray,
    theta_0: float,
    twist: np.ndarray,
    mach_r: np.ndarray,
    sig: np.ndarray,
    CT_req: float,
    airfoil_tables: list[AirfoilTable],
    r_boundaries: list[float],
    rotor: RotorConfig,
    warn_negative_lift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Compute converged spanwise inflow ratio distribution.

    Two-phase approach:
      1. Linear inflow (thin-airfoil Cl_alpha) with Prandtl tip loss — fast
         initial guess.
      2. Table-lookup iteration with relaxation until the momentum and blade-
         element inflow estimates agree.

    Returns
    -------
    lam      : inflow ratio,   shape (n_seg,)
    theta    : local pitch,    shape (n_seg,)  [theta_0 + twist]
    F        : tip-loss factor, shape (n_seg,)
    cl, cd, cm : aerodynamic coefficients, each shape (n_seg,)
    lambda_fail : True if the table-lookup loop did not converge
    """
    theta = theta_0 + twist
    lam = np.full(len(r), np.sqrt(max(CT_req, 0.0) / 2.0))

    # --- Phase 1: linear-theory inflow ---
    CL_alpha = 2.0 * np.pi
    for _ in range(500):
        F = _prandtl_tip_loss(r, lam, rotor.n_blades, rotor.tip_loss)
        arg = 1.0 + 32.0 * F * theta * r / (sig * CL_alpha)
        lam_new = sig * CL_alpha / (16.0 * F) * (np.sqrt(np.maximum(arg, 0.0)) - 1.0)
        if np.max(np.abs(lam_new - lam)) < 1e-5:
            lam = lam_new
            break
        lam = lam_new

    # --- Phase 2: table-lookup iteration with under-relaxation ---
    lambda_fail = False
    cl = cd = cm = np.zeros_like(r)

    for counter in range(50):
        F = _prandtl_tip_loss(r, lam, rotor.n_blades, rotor.tip_loss)
        cl, cd, cm = lookup_coefficients(
            r, theta, mach_r, lam, airfoil_tables, r_boundaries, warn_negative_lift
        )

        val = sig * cl * r / (8.0 * F)
        induced = np.sqrt(np.abs(val)) * np.sign(val)

        old_lam = lam.copy()
        lam = old_lam + 0.2 * (induced - old_lam)

        if np.sum(np.abs(old_lam - induced)) < 1e-5:
            break
        if counter == 49:
            lambda_fail = True

    return lam, theta, F, cl, cd, cm, lambda_fail


def _prandtl_tip_loss(
    r: np.ndarray,
    lam: np.ndarray,
    n_blades: int,
    use_tip_loss: bool,
) -> np.ndarray:
    """Compute Prandtl tip-loss factor F at each spanwise station."""
    if not use_tip_loss:
        return np.ones_like(r)
    phi = np.real(lam / r)
    with np.errstate(over="ignore", invalid="ignore"):
        # Overflow is benign: exp(+inf) → inf, clipped to 0.99 below
        f = np.exp(-n_blades / 2.0 * (1.0 - r) / (r * phi))
    f = np.clip(f, -0.99, 0.99)
    return np.real(2.0 / np.pi * np.arccos(f))
