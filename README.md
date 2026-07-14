# BEMT Rotor Analysis

Python implementation of **Blade Element Momentum Theory (BEMT)** for hovering helicopter rotors.

Developed at the University of Maryland.

## Features

- Prandtl tip-loss model (on/off)
- 2-D aerodynamic table lookup in (Mach, angle-of-attack) space
- Bisection method for collective pitch trim to a target thrust coefficient
- Linear blending between spanwise airfoil zones
- Sweep over rotor weight targets
- Clean dataclass-based API — easy to integrate into optimization loops

## Directory structure

```
BEMT_Python/
├── main.py              # CLI entry point
├── pipeline.py          # optional AGRC surrogate pipeline (requires TensorFlow)
├── bemt/
│   ├── solver.py        # BEMTSolver class (bisection trim + integration)
│   ├── airfoil.py       # AirfoilTable parser and AirfoilDatabase cache
│   ├── inflow.py        # spanwise inflow solver with Prandtl tip-loss
│   ├── lookup.py        # 2-D Mach×AoA coefficient interpolation
│   ├── config.py        # RotorConfig, BladeGeometry, AtmosphericConfig, ...
│   ├── io.py            # YAML config loader
│   ├── results.py       # BEMTResult dataclass
│   └── data/            # bundled C81 airfoil tables
├── examples/
│   ├── 01_basic_bemt.py
│   ├── 02_custom_c81.py
│   ├── 03_agrc_pipeline.py
│   └── example_config.yaml
└── tests/
    └── test_solver.py
```

## Installation

```bash
pip install -r requirements.txt
```

For the AGRC surrogate pipeline only:

```bash
pip install bemt-rotor[agrc]
```

Python 3.10+ recommended.

## Quick start

```bash
python main.py
```

Output for the default UH-60 case:

```
Weight(lb)          CT        FM         CPi         CP0   Coll(°)
------------------------------------------------------------------
CT=0.003621  FM=0.6458  Coll=15.48°  θ75=5.80°  CPi=0.000163  CP0=0.000075

Completed 1 case(s) in 0.15 s
```

Run a weight sweep:

```bash
python main.py --weights 2200 4000 6000
```

## Blade geometry: taper and twist

Chord (taper) and twist are specified as spanwise arrays in the YAML config:

```yaml
blade:
  r_stations: [0.19, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00]  # r/R
  chord_ft:   [1.20, 1.15, 1.10, 1.05, 1.00, 0.85, 0.70]  # tapered blade
  twist_deg:  [0.0, -2.5, -5.0, -7.5, -10.0, -12.5, -14.0]
```

You can provide as few or as many stations as you need — the solver interpolates them onto its internal integration grid. Via the Python API:

```python
from bemt.config import BladeGeometry
import numpy as np

blade = BladeGeometry(
    r_stations = list(np.linspace(0.19, 1.0, 20)),
    chord_ft   = list(np.linspace(1.2, 0.7, 20)),   # linearly tapered
    twist_deg  = list(np.linspace(0.0, -14.0, 20)),
)
```

## Airfoil input: two paths

### Path 1 — bring your own C81 table

Pass a C81 file directly via the CLI:

```bash
python main.py --c81 path/to/your_airfoil.dat --weights 2200 4000
```

Or reference it in the YAML config under `airfoils`:

```yaml
airfoils:
  MyAirfoil:
    source: c81
    file: path/to/your_airfoil.dat
```

### Path 2 — generate C81 from airfoil geometry via AGRC surrogate

If you have an airfoil coordinate file (x/y perimeter points), the AGRC surrogate neural network can predict the full Cl/Cd/Cm tables across Mach numbers:

```bash
# 1. Clone the surrogate model repo (one-time setup)
git clone https://github.com/anandaero747/AGRC-Surrogate

# 2. Run the pipeline
python pipeline.py airfoil.dat \
    --agrc-dir AGRC-Surrogate/agrc_surrogate \
    --weights 2200 4000
```

Or point to it from the YAML config:

```yaml
# After cloning https://github.com/anandaero747/AGRC-Surrogate
agrc_dir: AGRC-Surrogate/agrc_surrogate

airfoils:
  MyAirfoil:
    source: agrc
    geometry: airfoils/my_airfoil_profile.dat
```

> **Note:** When using the AGRC pipeline via the Python API (`airfoil_to_bemt`), rotor and blade geometry default to the UH-60 baseline (radius = 16.4 ft, 3 blades, constant chord, linear twist). To use a custom blade, pass `rotor_kwargs` and `blade_kwargs` explicitly:
>
> ```python
> results = airfoil_to_bemt(
>     "my_airfoil.dat",
>     agrc,
>     weights_lb=[2200, 4000],
>     rotor_kwargs={"radius": 20.0, "n_blades": 4, "tip_speed": 600.0},
>     blade_kwargs={
>         "r_stations": list(np.linspace(0.15, 1.0, 20)),
>         "chord_ft":   list(np.linspace(1.3, 0.8, 20)),
>         "twist_deg":  list(np.linspace(0.0, -12.0, 20)),
>     },
> )
> ```

## C81 airfoil table format

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

Bundled airfoils (usable via `source: builtin` in the YAML config):

| Name | Description |
|------|-------------|
| `SC1095_CFD` | SC1095 — 2D CFD |
| `SC1095_base` | SC1095 — baseline |
| `SC1095_TNN` | SC1095 — TNN model |
| `SC1095_Opt_TNN_Mach` | SC1095 — Mach-optimized TNN |
| `SC1095_CFD_pt_Baseline` | SC1095 CFD perturbation, baseline |
| `SC1095_CFD_pt_Case1` | SC1095 CFD perturbation, case 1 |
| `SC1095_CFD_pt_Case2` | SC1095 CFD perturbation, case 2 |
| `SC1095_NN_pt_Baseline` | SC1095 NN perturbation, baseline |
| `SC1095_NN_pt_Case1` | SC1095 NN perturbation, case 1 |
| `SC1095_NN_pt_Case2` | SC1095 NN perturbation, case 2 |
| `SC1095_Pt_Case2` | SC1095 perturbation, case 2 |
| `SC1095_CFD_Case2` | SC1095 CFD, case 2 |
| `Clark_Y` / `Clark_YH` | Clark Y variants |
| `MH60` | MH60 airfoil |
| `RC3_8` / `RC4_10` | RC-series airfoils |

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
