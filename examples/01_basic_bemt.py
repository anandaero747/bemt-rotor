"""Example 1: Basic hover analysis using the bundled SC1095 CFD airfoil tables.

Run from the repo root:
    python examples/01_basic_bemt.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bemt import (
    AtmosphericConfig,
    AirfoilDatabase,
    AirfoilZoneConfig,
    BladeGeometry,
    BEMTSolver,
    RotorConfig,
    WeightSweepConfig,
)

# ---------------------------------------------------------------------------
# Rotor geometry — UH-60 Black Hawk baseline
# ---------------------------------------------------------------------------
N_STATIONS = 40
ROOT_CUTOUT = 0.19

atmo = AtmosphericConfig(altitude=0.0, T_today=59.0)

rotor = RotorConfig(
    radius=16.4,        # ft
    n_blades=3,
    root_cutout=ROOT_CUTOUT,
    tip_speed=550.0,    # ft/s
    tip_loss=True,
)

blade = BladeGeometry(
    r_stations = list(np.linspace(ROOT_CUTOUT, 1.0, N_STATIONS)),
    chord_ft   = [1.08] * N_STATIONS,                              # uniform chord
    twist_deg  = list(np.linspace(0.0, -14.0, N_STATIONS)),       # linear wash-out
)

# Two zone boundaries both pointing to airfoil index 1 (SC1095 CFD)
az = AirfoilZoneConfig(airfoil_ids=["SC1095_CFD", "SC1095_CFD"],
                      r_boundaries=[ROOT_CUTOUT, 1.0])

# Load bundled airfoil tables (bemt/data/*.dat)
db = AirfoilDatabase()
tables = db.load_names(az.airfoil_ids)
airfoil_tables = [tables[name] for name in az.airfoil_ids]

solver = BEMTSolver(rotor, atmo, blade, az, airfoil_tables)

# ---------------------------------------------------------------------------
# Weight sweep
# ---------------------------------------------------------------------------
target_weights_lb = [2000, 2500, 3000, 4000, 5000, 6000]

print(f"{'Weight(lb)':>10}  {'CT':>10}  {'FM':>8}  {'CPi':>10}  {'CP0':>10}  {'Coll(°)':>8}")
print("-" * 66)

for w in target_weights_lb:
    CT_req = w / (atmo.rho * rotor.disk_area * rotor.tip_speed**2 * rotor.n_rotors)
    result = solver.solve(CT_req)

    if result.bisection_fail:
        print(f"{w:>10}  --- stalled ---")
    else:
        print(f"{w:>10}  {result.CT:>10.6f}  {result.FM:>8.4f}"
              f"  {result.CPi:>10.6f}  {result.CP0:>10.6f}  {result.collective_deg:>8.2f}")
