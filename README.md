# BEMT Rotor Analysis

Python implementation of **Blade Element Momentum Theory (BEMT)** for hovering helicopter rotors.

Ported from a MATLAB codebase developed at the University of Maryland.

## Features

- Prandtl tip-loss model (on/off)
- 2-D aerodynamic table lookup in (Mach, angle-of-attack) space
- Single and bi-taper chord distributions
- Bisection method for collective pitch trim to a target thrust coefficient
- Linear blending between spanwise airfoil zones
- Sweep over taper ratios and rotor weight targets
- Clean dataclass-based API — easy to integrate into optimization loops

## Directory structure

```
BEMT_Python/
├── main.py          # entry point; defines rotor configs and runs sweeps
├── config.py        # RotorConfig, AtmosphericConfig, TwistConfig, ...
├── airfoil.py       # AirfoilTable (parser) and AirfoilDatabase (cache)
├── lookup.py        # 2-D Mach×AoA coefficient interpolation
├── inflow.py        # spanwise inflow solver (linear + table-lookup)
├── solver.py        # BEMTSolver class (bisection trim + integration)
├── results.py       # BEMTResult dataclass
├── requirements.txt
└── airfoil/         # coefficient tables (.dat files)
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Quick start

```bash
python main.py
```

Output for the default UH-60 case:

```
CT=0.004729  FM=0.7631  Coll=9.12°  θ75=8.23°  CPi=0.000312  CP0=0.000097
Completed 1 case(s) in 0.42 s
```

## Changing the configuration

Edit `main.py` — the `uh60_config()` function returns all configuration objects.
To add a new rotor just write another function with the same signature and pass it to `run_sweep()`.

```python
atmo, rotor, twist, airfoil_zone, sweep = uh60_config()
results = run_sweep(atmo, rotor, twist, airfoil_zone, sweep,
                    airfoil_dir=Path("airfoil"))
```

## Airfoil data format

Tables are in PRASADUM-style `.dat` format:

```
SC1095: 2D CFD runs            ← description (skipped)
4 31    :nMach nAoA            ← grid dimensions
==============================
-- Lift coefficient
==============================
0.00  -10.0   -1.0697e+00     ← Mach  AoA_deg  Cl
...                            ← nMach × nAoA rows
(same pattern for Cd, then Cm)
```

Supported airfoil indices (set via `AirfoilZoneConfig.airfoil_indices`):

| Index | File |
|-------|------|
| 1 | SC1095\_CFD.dat |
| 2 | SC1095\_base.dat |
| 3 | SC1095\_TNN.dat |
| 4 | SC1095\_Opt\_TNN\_Mach.dat |
| 5–7 | SC1095\_CFD\_pt\_Baseline/Case1/Case2.dat |
| 8–10 | SC1095\_NN\_pt\_Baseline/Case1/Case2.dat |
| 11 | SC1095\_Pt\_Case2.dat |
| 12 | SC1095\_CFD\_Case2.dat |

## Output quantities (`BEMTResult`)

| Field | Description |
|-------|-------------|
| `CT` | Thrust coefficient |
| `CPi` | Induced power coefficient |
| `CP0` | Profile power coefficient |
| `FM` | Figure of merit |
| `kappa` | Induced power correction factor |
| `collective_deg` | Collective pitch, deg |
| `theta_75_deg` | Blade pitch at 75 % radius, deg |
| `mean_alpha_deg` | Span-averaged angle of attack, deg |
| `PL` | Power loading, lb/hp |
| `torque` | Rotor torque, lb·ft |
| `pitching_moment` | Total blade pitching moment, lb·ft |
