"""End-to-end pipeline: airfoil geometry → AGRC_Surrogate C81 → BEMT performance.

This module is an **optional** extension that requires TensorFlow + Keras.
Core BEMT functionality (``bemt`` package) has no such dependency.

Usage
-----
from pipeline import load_agrc_models, airfoil_to_bemt

agrc = load_agrc_models("/path/to/AGRC-Surrogate/agrc_surrogate")
# Clone the surrogate repo from: https://github.com/anandaero747/AGRC-Surrogate
results = airfoil_to_bemt("my_airfoil.dat", agrc, weights_lb=[2200, 4000])

CLI
---
python pipeline.py airfoil.dat --agrc-dir /path/to/agrc_surrogate --weights 2200 4000
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# Ensure the repo root is on sys.path so ``bemt`` is importable
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# Airfoil geometry helpers
# =============================================================================

def _read_xy(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(pts) < 20:
        raise ValueError(f"Too few coordinate points in {filepath}")
    xy = np.array(pts)
    return xy[:, 0], xy[:, 1]


def _rotate_to(x, y, idx):
    return np.concatenate([x[idx:], x[:idx]]), np.concatenate([y[idx:], y[:idx]])


def _interp_at(bx, by, xq=0.5):
    order = np.argsort(bx)
    xs, ys = bx[order], by[order]
    return float(np.interp(np.clip(xq, xs[0], xs[-1]), xs, ys))


def split_surfaces(filepath: str | Path,
                   te_tol: float = 1e-3,
                   min_pts: int = 8) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """Split a single-file airfoil perimeter into upper / lower surfaces."""
    x, y = _read_xy(filepath)
    keep = [0] + [i for i in range(1, len(x))
                  if abs(x[i]-x[i-1]) > 1e-12 or abs(y[i]-y[i-1]) > 1e-12]
    x, y = x[keep], y[keep]
    xmin, xmax = x.min(), x.max()
    chord = xmax - xmin
    x = (x - xmin) / chord
    y = y / chord

    te_mask = x >= 1.0 - te_tol
    idx_te = int(np.where(te_mask)[0][np.argmin(y[te_mask])]) if te_mask.any() \
             else int(np.argmax(x))
    x, y = _rotate_to(x, y, idx_te)

    idx_le = int(np.argmin(x))
    if idx_le in (0, len(x)-1):
        x, y = x[::-1], y[::-1]
        te_mask = x >= 1.0 - te_tol
        idx_te = int(np.where(te_mask)[0][np.argmin(y[te_mask])]) if te_mask.any() \
                 else int(np.argmax(x))
        x, y = _rotate_to(x, y, idx_te)
        idx_le = int(np.argmin(x))

    b1x, b1y = x[:idx_le+1], y[:idx_le+1]
    b2x, b2y = x[idx_le:],   y[idx_le:]
    if len(b1x) < min_pts or len(b2x) < min_pts:
        raise ValueError("Could not split airfoil into two valid branches.")

    if _interp_at(b1x, b1y) < _interp_at(b2x, b2y):
        return b2x, b2y, b1x, b1y
    return b1x, b1y, b2x, b2y


def _class_func(x):
    return np.sqrt(x) * (1 - x)


def cst_cheb_te(x, *a):
    *coeffs, te = a
    T = chebvander(2*x - 1, len(coeffs)-1)
    return _class_func(x) * (T @ np.array(coeffs)) + te * x


def fit_cst(xU, yU, xL, yL, n_terms: int = 10) -> np.ndarray:
    """Fit CST Chebyshev coefficients and return concatenated [upper | lower]."""
    pU, _ = curve_fit(cst_cheb_te, xU, yU, p0=np.zeros(n_terms))
    pL, _ = curve_fit(cst_cheb_te, xL, yL, p0=np.zeros(n_terms))
    return np.concatenate([pU, pL])[None, :]   # shape (1, 2*n_terms)


# =============================================================================
# AGRC model loading
# =============================================================================

class AGRCModels(NamedTuple):
    models:   dict   # tag → keras model
    pca:      dict   # "cl_pt3" → PCA object, etc.
    scalers:  dict   # "cl_pt3" → scaler, etc.
    scaler_x: object
    base_tables: dict  # "Cl" / "Cd" / "Cm" → DataFrame


def load_agrc_models(agrc_dir: str | Path) -> AGRCModels:
    """Load all AGRC neural-network models, PCA transformers, and base tables.

    Parameters
    ----------
    agrc_dir : path to the ``agrc_surrogate`` directory (contains ``assets/``).
    """
    import tensorflow as tf
    import pandas as pd

    agrc_dir = Path(agrc_dir)
    assets = agrc_dir / "assets"
    pre = assets / "preprocess"

    scaler_x = joblib.load(pre / "scaler_cst_v2.pkl")

    mach_tags = ["pt3", "pt4", "pt5", "pt6", "pt7", "pt8"]
    coefs     = ["cl", "cd", "cm"]
    models, pca, scalers = {}, {}, {}

    custom_objects = {"DTypePolicy": tf.keras.mixed_precision.Policy}
    for tag in mach_tags:
        models[tag] = tf.keras.models.load_model(
            str(assets / "models" / f"Forward_{tag}_non_opt_hyp_v2.h5"),
            custom_objects=custom_objects,
            compile=False)
        for coef in coefs:
            key = f"{coef}_{tag}"
            pca[key]     = joblib.load(pre / f"pca_{key}_v2.pkl")
            scalers[key] = joblib.load(pre / f"scaler_{key}_v2.pkl")

    data_dir = assets / "data"
    base_tables = {}
    for coef in ["Cl", "Cd", "Cm"]:
        df = _read_airfoil_dat(str(data_dir / f"{coef}_360_23012.dat"))
        base_tables[coef] = df

    print("AGRC models loaded.")
    return AGRCModels(models=models, pca=pca, scalers=scalers,
                      scaler_x=scaler_x, base_tables=base_tables)


def _read_airfoil_dat(filepath: str):
    import pandas as pd
    data = np.genfromtxt(filepath, filling_values=np.nan)
    data = data[~np.isnan(data).all(axis=1)]
    n_cols = data.shape[1]
    cols = ["AoA"] + [f"Mach_{round(0.1*i+0.1, 1)}" for i in range(1, n_cols)]
    return pd.DataFrame(data, columns=cols).sort_values("AoA").reset_index(drop=True)


# =============================================================================
# C81 table generation
# =============================================================================

def _blend(aoa_naca, data_naca, aoa_ann, data_ann,
           aoa_min, aoa_max, blend_width=10):
    aoa_full = np.linspace(-180, 180, 361)
    base = np.interp(aoa_full, aoa_naca, data_naca)
    ann  = np.interp(aoa_full, aoa_ann,  data_ann)
    out  = base.copy()

    idx  = (aoa_full >= aoa_min) & (aoa_full <= aoa_max)
    out[idx] = ann[idx]

    def _window(x, x0, w):
        return 0.5 * (1 - np.cos(np.pi * (x - x0 + w) / w))

    lo = (aoa_full >= aoa_min - blend_width) & (aoa_full < aoa_min)
    if lo.any():
        w = _window(aoa_full[lo], aoa_min, blend_width)
        out[lo] = (1-w)*base[lo] + w*ann[lo]

    hi = (aoa_full > aoa_max) & (aoa_full <= aoa_max + blend_width)
    if hi.any():
        w = 1 - _window(aoa_full[hi], aoa_max, blend_width)
        out[hi] = (1-w)*ann[hi] + w*base[hi]

    return aoa_full, out


def predict_c81(agrc: AGRCModels, X_scaled: np.ndarray):
    """Run AGRC prediction and return an AirfoilTable (no file I/O)."""
    from bemt.airfoil import AirfoilTable

    mach_tags  = ["pt3","pt4","pt5","pt6","pt7","pt8"]
    mach_cols  = ["Mach_0.3","Mach_0.4","Mach_0.5","Mach_0.6","Mach_0.7","Mach_0.8"]
    aoa_ann    = np.linspace(-10, 20, 31)
    aoa_ann_hi = np.linspace(-4,  20, 25)

    Cl_cols, Cd_cols, Cm_cols = [], [], []

    for tag, mcol in zip(mach_tags, mach_cols):
        pred = agrc.models[tag].predict(X_scaled, verbose=0)

        def _inv(coef, s, e):
            raw = np.array(pred[:, s:e])
            out = agrc.scalers[f"{coef}_{tag}"].inverse_transform(
                      agrc.pca[f"{coef}_{tag}"].inverse_transform(raw))
            if coef == "cd":
                out = np.exp(out)
            return out.flatten()

        cl_p = _inv("cl",  0, 10)
        cd_p = _inv("cd", 10, 28)
        cm_p = _inv("cm", 28, 36)

        aoa_a = aoa_ann_hi if tag in ("pt7","pt8") else aoa_ann
        aoa_min = -4 if tag in ("pt7","pt8") else -10

        df_cl = agrc.base_tables["Cl"]
        df_cd = agrc.base_tables["Cd"]
        df_cm = agrc.base_tables["Cm"]

        _, cl_b = _blend(df_cl["AoA"].values, df_cl[mcol].values, aoa_a, cl_p, aoa_min, 20)
        _, cd_b = _blend(df_cd["AoA"].values, df_cd[mcol].values, aoa_a, cd_p, aoa_min, 20)
        _, cm_b = _blend(df_cm["AoA"].values, df_cm[mcol].values, aoa_a, cm_p, aoa_min, 20)

        Cl_cols.append(np.round(cl_b, 4))
        Cd_cols.append(np.round(cd_b, 4))
        Cm_cols.append(np.round(cm_b, 4))

    # Duplicate M0.3 for the M0.1 and M0.2 placeholder columns
    Cl_mat = np.column_stack([Cl_cols[0], Cl_cols[0]] + Cl_cols)
    Cd_mat = np.column_stack([Cd_cols[0], Cd_cols[0]] + Cd_cols)
    Cm_mat = np.column_stack([Cm_cols[0], Cm_cols[0]] + Cm_cols)

    aoa_full   = np.linspace(-180, 180, 361)
    mach_vals  = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    return AirfoilTable(
        mach_vals  = mach_vals,
        alpha_vals = aoa_full,
        cl = Cl_mat.T,   # (n_mach, n_alpha)
        cd = Cd_mat.T,
        cm = Cm_mat.T,
    )


# =============================================================================
# Full pipeline
# =============================================================================

def airfoil_to_bemt(
    airfoil_dat: str | Path,
    agrc: AGRCModels,
    weights_lb: list[float] | None = None,
    rotor_kwargs: dict | None = None,
    blade_kwargs: dict | None = None,
) -> list:
    """One-call pipeline: airfoil .dat → C81 → BEMT results.

    Parameters
    ----------
    airfoil_dat  : path to airfoil coordinate file (full perimeter .dat)
    agrc         : loaded AGRC models from load_agrc_models()
    weights_lb   : list of rotor weights in lb to solve for
    rotor_kwargs : optional overrides for RotorConfig fields
    blade_kwargs : optional overrides for BladeGeometry fields

    Returns
    -------
    list of BEMTResult
    """
    from bemt.config import (AtmosphericConfig, RotorConfig,
                              BladeGeometry, AirfoilZoneConfig)
    from bemt.solver import BEMTSolver

    if weights_lb is None:
        weights_lb = [2200.0]

    xU, yU, xL, yL = split_surfaces(airfoil_dat)
    xU, yU = xU[np.argsort(xU)], yU[np.argsort(xU)]
    xL, yL = xL[np.argsort(xL)], yL[np.argsort(xL)]
    X_input  = fit_cst(xU, yU, xL, yL)
    X_scaled = agrc.scaler_x.transform(X_input)

    table = predict_c81(agrc, X_scaled)

    atmo  = AtmosphericConfig()
    ROOT  = 0.19
    N     = 40

    rk = rotor_kwargs or {}
    rotor = RotorConfig(radius=rk.get("radius", 16.4),
                        n_blades=rk.get("n_blades", 3),
                        root_cutout=ROOT,
                        tip_speed=rk.get("tip_speed", 550.0))

    bk = blade_kwargs or {}
    blade = BladeGeometry(
        r_stations = bk.get("r_stations", list(np.linspace(ROOT, 1.0, N))),
        chord_ft   = bk.get("chord_ft",   [1.08] * N),
        twist_deg  = bk.get("twist_deg",  list(np.linspace(0.0, -14.0, N))),
    )

    az     = AirfoilZoneConfig(airfoil_ids=["agrc_table", "agrc_table"],
                               r_boundaries=[ROOT, 1.0])
    solver = BEMTSolver(rotor, atmo, blade, az, [table, table])

    results = []
    for w in weights_lb:
        CT_req = w / (atmo.rho * rotor.disk_area * rotor.tip_speed**2)
        result = solver.solve(CT_req)
        results.append(result)

    return results


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Airfoil geometry → AGRC C81 → BEMT performance")
    parser.add_argument("airfoil", help="Path to airfoil .dat file")
    parser.add_argument(
        "--agrc-dir", required=True, metavar="DIR",
        help="Path to the agrc_surrogate directory (contains assets/)")
    parser.add_argument("--weights", nargs="+", type=float,
                        default=[2200.0], metavar="LB",
                        help="Rotor weight(s) in lb (default: 2200)")
    args = parser.parse_args()

    print("Loading AGRC models…")
    agrc = load_agrc_models(args.agrc_dir)

    print(f"Running pipeline for {args.airfoil}…")
    results = airfoil_to_bemt(args.airfoil, agrc, weights_lb=args.weights)

    print(f"\n{'Weight(lb)':>10}  {'CT':>10}  {'FM':>8}  {'CP':>10}  {'Coll(°)':>8}")
    print("-" * 56)
    for w, r in zip(args.weights, results):
        if r.bisection_fail:
            print(f"{w:>10.0f}  {'--- rotor stalled (collective limit reached) ---'}")
        else:
            print(f"{w:>10.0f}  {r.CT:>10.6f}  {r.FM:>8.4f}"
                  f"  {r.CPi+r.CP0:>10.6f}  {r.collective_deg:>8.2f}")
