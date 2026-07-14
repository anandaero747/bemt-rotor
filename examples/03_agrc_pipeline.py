"""Example 3: Full pipeline — airfoil geometry → AGRC surrogate → BEMT.

Requirements (beyond numpy/scipy):
    pip install bemt-rotor[agrc]
    # or: pip install tensorflow keras joblib scikit-learn pandas

The AGRC_Surrogate models must be cloned separately:
    git clone https://github.com/anandaero747/AGRC-Surrogate
    Then pass the path to the agrc_surrogate/ subdirectory with --agrc-dir.

Run from the repo root:
    python examples/03_agrc_pipeline.py  \\
        airfoil.dat                       \\
        --agrc-dir /path/to/agrc_surrogate \\
        --weights 2200 4000 6000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Airfoil .dat → AGRC C81 → BEMT hover performance")
    parser.add_argument("airfoil", type=Path,
                        help="Airfoil coordinate file (full perimeter, .dat format)")
    parser.add_argument("--agrc-dir", required=True, type=Path, metavar="DIR",
                        help="Path to the agrc_surrogate directory (must contain assets/)")
    parser.add_argument("--weights", nargs="+", type=float,
                        default=[2200.0], metavar="LB",
                        help="Rotor weight(s) in lb (default: 2200)")
    args = parser.parse_args()

    try:
        from pipeline import load_agrc_models, airfoil_to_bemt
    except ImportError as exc:
        print("ERROR: pipeline.py dependencies not installed.")
        print("  Install with:  pip install bemt-rotor[agrc]")
        print(f"  Details: {exc}")
        sys.exit(1)

    print("Loading AGRC surrogate models…")
    agrc = load_agrc_models(args.agrc_dir)

    print(f"Running pipeline for {args.airfoil}…")
    results = airfoil_to_bemt(args.airfoil, agrc, weights_lb=args.weights)

    print(f"\n{'Weight(lb)':>10}  {'CT':>10}  {'FM':>8}  {'CP':>10}  {'Coll(°)':>8}")
    print("-" * 56)
    for w, r in zip(args.weights, results):
        if r.bisection_fail:
            print(f"{w:>10.0f}  --- rotor stalled (collective limit reached) ---")
        else:
            print(f"{w:>10.0f}  {r.CT:>10.6f}  {r.FM:>8.4f}"
                  f"  {r.CPi + r.CP0:>10.6f}  {r.collective_deg:>8.2f}")


if __name__ == "__main__":
    main()
