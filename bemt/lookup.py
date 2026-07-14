#Spanwise airfoil coefficient lookup via 2-D (Mach, AoA) interpolation.
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .airfoil import AirfoilTable


def lookup_coefficients(
    r: np.ndarray,
    theta: np.ndarray,
    mach_r: np.ndarray,
    lam: np.ndarray,
    airfoil_tables: list[AirfoilTable],
    r_boundaries: list[float],
    warn_negative_lift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = lam / r
    alpha_deg = np.rad2deg(theta - phi)

    n_zones = len(airfoil_tables)
    n_seg = len(r)
    cl1 = np.zeros((n_zones, n_seg))
    cd1 = np.zeros((n_zones, n_seg))
    cm1 = np.zeros((n_zones, n_seg))

    for l, table in enumerate(airfoil_tables):
        # Reuse previous result when the same airfoil object appears consecutively
        if l > 0 and airfoil_tables[l] is airfoil_tables[l - 1]:
            cl1[l] = cl1[l - 1]
            cd1[l] = cd1[l - 1]
            cm1[l] = cm1[l - 1]
            continue

        alpha_clamped = np.clip(alpha_deg, table.alpha_vals.min(), table.alpha_vals.max())
        query = np.column_stack([mach_r, alpha_clamped])

        cl1[l] = _interp2d(table.mach_vals, table.alpha_vals, table.cl, query)
        cd1[l] = _interp2d(table.mach_vals, table.alpha_vals, table.cd, query)
        cm1[l] = _interp2d(table.mach_vals, table.alpha_vals, table.cm, query)

    cl = np.zeros(n_seg)
    cd = np.zeros(n_seg)
    cm = np.zeros(n_seg)
    n_warn = 0

    for k in range(n_seg):
        ind = _zone_index(r[k], r_boundaries)
        ratio = (r[k] - r_boundaries[ind]) / (r_boundaries[ind + 1] - r_boundaries[ind])
        w0, w1 = 1.0 - ratio, ratio

        cl[k] = w0 * cl1[ind, k] + w1 * cl1[ind + 1, k]
        cd[k] = w0 * cd1[ind, k] + w1 * cd1[ind + 1, k]
        cm[k] = w0 * cm1[ind, k] + w1 * cm1[ind + 1, k]

        if cl[k] < 0.0:
            cl[k] = 0.0
            n_warn += 1

    if n_warn > 0 and warn_negative_lift:
        import warnings
        warnings.warn(f"Negative lift clipped at {n_warn} spanwise stations.")

    return cl, cd, cm


def _interp2d(
    mach_vals: np.ndarray,
    alpha_vals: np.ndarray,
    data: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    interp = RegularGridInterpolator(
        (mach_vals, alpha_vals),
        data,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return interp(query)


def _zone_index(r_val: float, boundaries: list[float]) -> int:
    for j in range(len(boundaries) - 1):
        if boundaries[j] <= r_val <= boundaries[j + 1]:
            return j
    return len(boundaries) - 2
