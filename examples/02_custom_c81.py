"""Example 2: Use your own C81 airfoil table instead of the bundled tables.

This demonstrates how any user can plug in their own airfoil data without
depending on the AGRC surrogate or any specific database.

The C81 format expected::

    C81 Table

    Cl vs alpha & Mach
    alpha<TAB>M0.1<TAB>M0.2<TAB>...
    -180.0<TAB>0.04<TAB>0.04<TAB>...
    ...

    Cd vs alpha & Mach
    ...

    Cm vs alpha & Mach
    ...

Run from the repo root:
    python examples/02_custom_c81.py --c81 path/to/your_airfoil.dat
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bemt import (
    AtmosphericConfig,
    AirfoilTable,
    AirfoilZoneConfig,
    BladeGeometry,
    BEMTSolver,
    RotorConfig,
)


def main(c81_path: Path) -> None:
    N_STATIONS = 40
    ROOT_CUTOUT = 0.19

    atmo = AtmosphericConfig(altitude=0.0, T_today=59.0)
    rotor = RotorConfig(
        radius=16.4, n_blades=3,
        root_cutout=ROOT_CUTOUT, tip_speed=550.0,
    )
    blade = BladeGeometry(
        r_stations = list(np.linspace(ROOT_CUTOUT, 1.0, N_STATIONS)),
        chord_ft   = [1.08] * N_STATIONS,
        twist_deg  = list(np.linspace(0.0, -14.0, N_STATIONS)),
    )
    az = AirfoilZoneConfig(airfoil_ids=["custom", "custom"],
                           r_boundaries=[ROOT_CUTOUT, 1.0])

    print(f"Loading C81 table from {c81_path} …")
    table = AirfoilTable.from_c81(c81_path)
    print(f"  Mach range: {table.mach_vals[0]:.2f} – {table.mach_vals[-1]:.2f}"
          f"  ({table.n_mach} values)")
    print(f"  AoA  range: {table.alpha_vals[0]:.1f}° – {table.alpha_vals[-1]:.1f}°"
          f"  ({table.n_alpha} points)")

    airfoil_tables = [table, table]
    solver = BEMTSolver(rotor, atmo, blade, az, airfoil_tables)

    target_weights_lb = [2200, 4000, 6000]
    print(f"\n{'Weight(lb)':>10}  {'CT':>10}  {'FM':>8}  {'Coll(°)':>8}")
    print("-" * 46)

    for w in target_weights_lb:
        CT_req = w / (atmo.rho * rotor.disk_area * rotor.tip_speed**2)
        result = solver.solve(CT_req)
        if result.bisection_fail:
            print(f"{w:>10}  --- stalled ---")
        else:
            print(f"{w:>10}  {result.CT:>10.6f}  {result.FM:>8.4f}"
                  f"  {result.collective_deg:>8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEMT with a custom C81 table")
    parser.add_argument("--c81", required=True, metavar="FILE", type=Path,
                        help="Path to C81-format airfoil data file")
    args = parser.parse_args()
    main(args.c81)
