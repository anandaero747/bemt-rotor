"""Output data structure for a single BEMT solve."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BEMTResult:
    """Integrated performance quantities from one BEMT evaluation."""

    collective_deg: float   # collective pitch offset, deg
    theta_75_deg: float     # blade pitch at 75% radius, deg
    mean_alpha_deg: float   # span-averaged angle of attack, deg
    CT: float               # thrust coefficient
    CPi: float              # induced power coefficient
    CP0: float              # profile power coefficient
    FM: float               # figure of merit
    kappa: float            # induced power correction factor
    PL: float               # power loading, lb/hp
    torque: float           # rotor torque, lb·ft
    pitching_moment: float  # total blade pitching moment, lb·ft
    vtip: float             # tip speed, ft/s
    bisection_fail: bool    # collective bisection did not converge
    lambda_fail: bool       # inflow iteration did not converge

    # Spanwise arrays — one value per blade element (length = n_segments)
    r_span: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_span_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    cl_span: np.ndarray = field(default_factory=lambda: np.array([]))
    cd_span: np.ndarray = field(default_factory=lambda: np.array([]))
    dCT_span: np.ndarray = field(default_factory=lambda: np.array([]))
    dCPi_span: np.ndarray = field(default_factory=lambda: np.array([]))
    dCP0_span: np.ndarray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        return (
            f"CT={self.CT:.6f}  FM={self.FM:.4f}  "
            f"Coll={self.collective_deg:.2f}°  θ75={self.theta_75_deg:.2f}°  "
            f"CPi={self.CPi:.6f}  CP0={self.CP0:.6f}"
        )
