"""Physics sanity checks for the BEMT solver."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on the path when running tests from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bemt import (
    AtmosphericConfig,
    AirfoilDatabase,
    AirfoilZoneConfig,
    BladeGeometry,
    BEMTSolver,
    RotorConfig,
)

_AIRFOIL_NAME = "SC1095_CFD"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uh60():
    """Minimal UH-60-like rotor with bundled SC1095 CFD tables."""
    N, ROOT = 40, 0.19
    atmo  = AtmosphericConfig()
    rotor = RotorConfig(radius=16.4, n_blades=3, root_cutout=ROOT, tip_speed=550.0)
    blade = BladeGeometry(
        r_stations = list(np.linspace(ROOT, 1.0, N)),
        chord_ft   = [1.08] * N,
        twist_deg  = list(np.linspace(0.0, -14.0, N)),
    )
    az = AirfoilZoneConfig(airfoil_ids=[_AIRFOIL_NAME, _AIRFOIL_NAME], r_boundaries=[ROOT, 1.0])
    db = AirfoilDatabase()
    tables = db.load_names(az.airfoil_ids)
    airfoil_tables = [tables[name] for name in az.airfoil_ids]

    solver = BEMTSolver(rotor, atmo, blade, az, airfoil_tables)
    return solver, atmo, rotor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ct_matches_target(uh60):
    """Solved CT must be within 1e-4 of the requested value."""
    solver, atmo, rotor = uh60
    target_weight_lb = 2200.0
    CT_req = target_weight_lb / (atmo.rho * rotor.disk_area * rotor.tip_speed**2)
    result = solver.solve(CT_req)

    assert not result.bisection_fail, "Bisection should converge for a realistic CT"
    assert abs(result.CT - CT_req) / CT_req < 1e-3


def test_fm_physical_range(uh60):
    """Figure of Merit must be between 0 and 1."""
    solver, atmo, rotor = uh60
    CT_req = 2200.0 / (atmo.rho * rotor.disk_area * rotor.tip_speed**2)
    result = solver.solve(CT_req)

    assert 0.0 < result.FM <= 1.0


def test_higher_weight_higher_collective(uh60):
    """Heavier rotor needs more collective pitch to generate more thrust."""
    solver, atmo, rotor = uh60
    base = atmo.rho * rotor.disk_area * rotor.tip_speed**2

    r1 = solver.solve(2000.0 / base)
    r2 = solver.solve(4000.0 / base)

    assert r2.collective_deg > r1.collective_deg


def test_higher_weight_higher_power(uh60):
    """Induced power coefficient scales with thrust (CT)."""
    solver, atmo, rotor = uh60
    base = atmo.rho * rotor.disk_area * rotor.tip_speed**2

    r1 = solver.solve(2000.0 / base)
    r2 = solver.solve(4000.0 / base)

    assert r2.CPi > r1.CPi


def test_no_tip_loss_higher_fm():
    """Disabling Prandtl tip-loss should yield optimistic (higher) FM."""
    N, ROOT = 40, 0.19
    atmo = AtmosphericConfig()
    blade = BladeGeometry(
        r_stations = list(np.linspace(ROOT, 1.0, N)),
        chord_ft   = [1.08] * N,
        twist_deg  = list(np.linspace(0.0, -14.0, N)),
    )
    az = AirfoilZoneConfig(airfoil_ids=[_AIRFOIL_NAME, _AIRFOIL_NAME], r_boundaries=[ROOT, 1.0])
    db = AirfoilDatabase()
    tables = db.load_names(az.airfoil_ids)
    airfoil_tables = [tables[name] for name in az.airfoil_ids]

    rotor_with    = RotorConfig(radius=16.4, n_blades=3, root_cutout=ROOT,
                                tip_speed=550.0, tip_loss=True)
    rotor_without = RotorConfig(radius=16.4, n_blades=3, root_cutout=ROOT,
                                tip_speed=550.0, tip_loss=False)

    CT_req = 2200.0 / (atmo.rho * rotor_with.disk_area * rotor_with.tip_speed**2)

    fm_with    = BEMTSolver(rotor_with,    atmo, blade, az, airfoil_tables).solve(CT_req).FM
    fm_without = BEMTSolver(rotor_without, atmo, blade, az, airfoil_tables).solve(CT_req).FM

    assert fm_without >= fm_with


def test_blade_geometry_validation():
    """BladeGeometry must reject mismatched array lengths."""
    with pytest.raises(ValueError, match="same length"):
        BladeGeometry(r_stations=[0.2, 0.5, 1.0],
                      chord_ft=[1.0, 1.0],        # wrong length
                      twist_deg=[0.0, -5.0, -10.0])


def test_blade_geometry_monotonic():
    """BladeGeometry must reject non-monotonic r_stations."""
    with pytest.raises(ValueError, match="strictly increasing"):
        BladeGeometry(r_stations=[0.2, 0.8, 0.5],
                      chord_ft=[1.0, 1.0, 1.0],
                      twist_deg=[0.0, -5.0, -10.0])
