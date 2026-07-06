"""BEMTSolver: bisection-based collective trim and performance integration."""
from __future__ import annotations

import numpy as np

from .airfoil import AirfoilTable
from .config import AtmosphericConfig, AirfoilZoneConfig, BladeGeometry, RotorConfig
from .inflow import compute_inflow
from .results import BEMTResult


class BEMTSolver:
    """Blade Element Momentum Theory solver for a hovering rotor.

    Chord and twist are supplied as explicit spanwise arrays via BladeGeometry
    and interpolated onto the internal BEMT grid.

    Usage
    -----
    solver = BEMTSolver(rotor, atmo, blade_geom, airfoil_zone, airfoil_tables)
    result = solver.solve(CT_req=0.005)
    """

    def __init__(
        self,
        rotor: RotorConfig,
        atmo: AtmosphericConfig,
        blade_geom: BladeGeometry,
        airfoil_zone: AirfoilZoneConfig,
        airfoil_tables: list[AirfoilTable],
    ) -> None:
        self.rotor = rotor
        self.atmo = atmo
        self.blade_geom = blade_geom
        self.airfoil_zone = airfoil_zone
        self.airfoil_tables = airfoil_tables
        self._build_grid()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, CT_req: float) -> BEMTResult:
        """Trim collective pitch to hit CT_req and return integrated performance."""
        theta_0, bisection_fail = self._bisect_collective(CT_req)

        lam, theta, F, cl, cd, cm, lambda_fail = compute_inflow(
            self._r, theta_0, self._twist_rad, self._mach_r,
            self._sig, CT_req, self.airfoil_tables,
            self.airfoil_zone.r_boundaries, self.rotor,
            warn_negative_lift=True,
        )

        return self._integrate(theta_0, lam, theta, F, cl, cd, cm,
                               bisection_fail, lambda_fail)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_grid(self) -> None:
        """Interpolate blade geometry onto the uniform BEMT segment grid."""
        rc = self.rotor.root_cutout
        R = self.rotor.radius
        n = self.rotor.n_segments

        dy = (1.0 - rc) * R / n
        self._dr = dy / R
        i = np.arange(1, n + 1)
        self._r = rc + (i - 0.5) * self._dr          # segment midpoints, r/R
        self._mach_r = self.rotor.tip_speed / self.atmo.aspeed * self._r

        r_in = np.asarray(self.blade_geom.r_stations)
        self._twist_rad = np.interp(self._r, r_in,
                                    np.deg2rad(self.blade_geom.twist_deg))
        chord = np.interp(self._r, r_in,
                          np.asarray(self.blade_geom.chord_ft))
        self._sig = self.rotor.n_blades * chord / (np.pi * R)
        self._chord = chord

    def _bisect_collective(self, CT_req: float) -> tuple[float, bool]:
        """Find collective pitch that satisfies CT_req via bisection."""
        th_lo = np.deg2rad(0.0)
        th_hi = np.deg2rad(50.0)
        bisection_fail = False

        for iteration in range(50):
            th_mid = 0.5 * (th_lo + th_hi)
            lam, _, F, _, _, _, _ = compute_inflow(
                self._r, th_mid, self._twist_rad, self._mach_r,
                self._sig, CT_req, self.airfoil_tables,
                self.airfoil_zone.r_boundaries, self.rotor,
            )
            CT = np.sum(4.0 * F * self._r * lam**2 * self._dr)
            err = CT_req - CT

            if abs(err) <= 1e-7:
                return th_mid, bisection_fail

            if CT >= CT_req:
                th_hi = th_mid
            else:
                th_lo = th_mid

            if iteration == 49:
                bisection_fail = True

        return 0.5 * (th_lo + th_hi), bisection_fail

    def _integrate(
        self,
        theta_0: float,
        lam: np.ndarray,
        theta: np.ndarray,
        F: np.ndarray,
        cl: np.ndarray,
        cd: np.ndarray,
        cm: np.ndarray,
        bisection_fail: bool,
        lambda_fail: bool,
    ) -> BEMTResult:
        r = self._r
        dr = self._dr
        sig = self._sig
        chord = self._chord
        R = self.rotor.radius
        rho = self.atmo.rho
        A = self.rotor.disk_area
        vtip = self.rotor.tip_speed
        rc = self.rotor.root_cutout

        CT_arr = 4.0 * F * r * lam**2 * dr
        CPi_arr = CT_arr * lam
        CP0_arr = 0.5 * sig * cd * r**3 * dr

        CT_total = CT_arr.sum()
        CPi = CPi_arr.sum()
        CP0 = CP0_arr.sum()
        CP = CPi + CP0

        FM = (CT_total**1.5 / np.sqrt(2.0)) / CP if CP > 0 else 0.0
        kappa = CPi * np.sqrt(2.0) / CT_total**1.5 if CT_total > 0 else 0.0

        dy = (1.0 - rc) * R / self.rotor.n_segments
        sec_vel = self._mach_r * self.atmo.aspeed
        dim_cm = 0.5 * rho * sec_vel**2 * (chord * dy) * cm * chord
        total_cm = dim_cm.sum()

        alpha = theta - lam / r
        th75 = float(np.interp(0.75, r, theta)) * 180.0 / np.pi

        return BEMTResult(
            collective_deg=float(np.rad2deg(theta_0)),
            theta_75_deg=th75,
            mean_alpha_deg=float(np.rad2deg(alpha.mean())),
            CT=CT_total,
            CPi=CPi,
            CP0=CP0,
            FM=FM,
            kappa=kappa,
            PL=CT_total / CP / vtip * 550.0 if CP > 0 else 0.0,
            torque=CP * rho * A * vtip**2 * R,
            pitching_moment=total_cm,
            vtip=vtip,
            bisection_fail=bisection_fail,
            lambda_fail=lambda_fail,
        )
