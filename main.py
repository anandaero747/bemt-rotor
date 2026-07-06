"""Entry point: configure, sweep, and print BEMT results.

Usage
-----
# Run with a YAML config file (recommended):
    python main.py --input examples/example_config.yaml

# Run the built-in UH-60 baseline with bundled SC1095 tables:
    python main.py

# Use a custom C81 table directly:
    python main.py --c81 path/to/your_airfoil_C81.dat --weights 2200 4000

# Override target weights on any run:
    python main.py --input my_config.yaml --weights 2200 4000 6000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from bemt.config import (
    AtmosphericConfig,
    AirfoilZoneConfig,
    BladeGeometry,
    RotorConfig,
    WeightSweepConfig,
)
from bemt.airfoil import AirfoilDatabase, AirfoilResolver, AirfoilTable
from bemt.results import BEMTResult
from bemt.solver import BEMTSolver


# ---------------------------------------------------------------------------
# Built-in UH-60 baseline (used when no --input is given)
# ---------------------------------------------------------------------------

def uh60_config() -> tuple[AtmosphericConfig, RotorConfig, BladeGeometry,
                            AirfoilZoneConfig, WeightSweepConfig]:
    """UH-60 Black Hawk baseline hovering configuration."""
    N, ROOT = 40, 0.19
    atmo  = AtmosphericConfig(altitude=0.0, T_today=59.0)
    rotor = RotorConfig(radius=16.4, n_blades=3, n_rotors=1,
                        root_cutout=ROOT, tip_speed=550.0, tip_loss=True)
    blade = BladeGeometry(
        r_stations = list(np.linspace(ROOT, 1.0, N)),
        chord_ft   = [1.08] * N,
        twist_deg  = list(np.linspace(0.0, -14.0, N)),
    )
    az    = AirfoilZoneConfig(airfoil_ids=["SC1095_CFD", "SC1095_CFD"],
                              r_boundaries=[ROOT, 1.0])
    sweep = WeightSweepConfig(target_weights_lb=[2200.0])
    return atmo, rotor, blade, az, sweep


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(
    atmo: AtmosphericConfig,
    rotor: RotorConfig,
    blade: BladeGeometry,
    az: AirfoilZoneConfig,
    sweep: WeightSweepConfig,
    airfoil_tables: list[AirfoilTable],
    weights_lb: list[float] | None = None,
) -> list[BEMTResult]:
    """Run one BEMT solve per target weight."""
    solver = BEMTSolver(rotor, atmo, blade, az, airfoil_tables)
    target = weights_lb if weights_lb is not None else sweep.target_weights_lb

    results: list[BEMTResult] = []
    for w in target:
        CT_req = w / (atmo.rho * rotor.disk_area * rotor.tip_speed**2 * rotor.n_rotors)
        result = solver.solve(CT_req)
        results.append(result)
        print(result)

    return results


# ---------------------------------------------------------------------------
# Airfoil table builder
# ---------------------------------------------------------------------------

def build_airfoil_tables(
    az: AirfoilZoneConfig,
    airfoil_defs: dict | None = None,
    c81_dir: Path | None = None,
    agrc_dir: Path | None = None,
    c81_path: Path | None = None,
) -> list[AirfoilTable]:
    """Resolve airfoil zone IDs to AirfoilTable objects.

    Priority order:
    1. ``c81_path`` — single C81 file applied to every zone (quick override)
    2. ``airfoil_defs`` — per-airfoil source definitions (from YAML)
    3. Fallback — treat each airfoil_id as a bundled PRASADUM table name
    """
    if c81_path is not None:
        table = AirfoilTable.from_c81(c81_path)
        return [table] * len(az.airfoil_ids)

    if airfoil_defs is not None:
        resolver = AirfoilResolver(airfoil_defs,
                                   c81_dir=c81_dir,
                                   agrc_dir=agrc_dir)
        return resolver.resolve_zone(az.airfoil_ids)

    # Fallback: treat IDs as bundled PRASADUM table names
    db = AirfoilDatabase()
    tables_map = db.load_names(az.airfoil_ids)
    return [tables_map[name] for name in az.airfoil_ids]


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def _print_header():
    print(f"\n{'Weight(lb)':>10}  {'CT':>10}  {'FM':>8}  "
          f"{'CPi':>10}  {'CP0':>10}  {'Coll(°)':>8}")
    print("-" * 66)


def main() -> list[BEMTResult]:
    parser = argparse.ArgumentParser(
        description="BEMT hover performance solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument(
        "--input", "-i", metavar="CONFIG.yaml", type=Path, default=None,
        help="YAML configuration file (rotor, blade, airfoil definitions, sweep)")
    parser.add_argument(
        "--c81", metavar="FILE", type=Path, default=None,
        help="Single C81 airfoil table applied to all zones (overrides YAML airfoils)")
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None, metavar="LB",
        help="Override target weight(s) in lb")
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.input is not None:
        from bemt.io import load_config
        atmo, rotor, blade, az, sweep, airfoil_defs, c81_dir, agrc_dir = \
            load_config(args.input)
        airfoil_tables = build_airfoil_tables(
            az,
            airfoil_defs=airfoil_defs,
            c81_dir=c81_dir,
            agrc_dir=agrc_dir,
            c81_path=args.c81,
        )
    else:
        atmo, rotor, blade, az, sweep = uh60_config()
        airfoil_tables = build_airfoil_tables(az, c81_path=args.c81)

    _print_header()
    results = run_sweep(atmo, rotor, blade, az, sweep,
                        airfoil_tables, weights_lb=args.weights)

    print(f"\nCompleted {len(results)} case(s) in {time.perf_counter() - t0:.3f} s")
    return results


if __name__ == "__main__":
    main()
