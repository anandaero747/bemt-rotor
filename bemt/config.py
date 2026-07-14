#Configuration dataclasses for rotor, atmosphere, blade geometry, airfoil zones, and sweep.
import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class AtmosphericConfig:
    """ISA-style atmospheric conditions at a given altitude."""

    altitude: float = 0.0         # ft
    T_today: float = 59.0         # surface temperature, deg F
    rho: float = 0.0023769        # air density, slug/ft^3
    aspeed: float = 1116.37139    # speed of sound, ft/s

    def __post_init__(self) -> None:
        temp_F = self.T_today - 0.00375 * self.altitude
        temp_K = 273.0 + (temp_F - 32.0) * 5.0 / 9.0
        vis_si = 1.512041288 * temp_K**1.5 / (temp_K + 120.0) * 1e-6  # Pa·s
        self.vis: float = vis_si * 0.0685218 / 3.28084  # slug/(ft·s)


@dataclass
class RotorConfig:
    radius: float = 16.4           # ft
    n_blades: int = 3
    n_rotors: int = 1
    root_cutout: float = 0.19      # r/R
    tip_speed: float = 550.0       # ft/s — ignored when rpm is given
    rpm: float | None = None       # if set, tip_speed = rpm * radius * pi / 30
    tip_loss: bool = True          # Prandtl tip-loss correction

    # Derived quantities — set in __post_init__
    disk_area: float = field(init=False)
    n_segments: int = field(init=False)   # number of spanwise blade elements

    def __post_init__(self) -> None:
        if self.rpm is not None:
            self.tip_speed = self.rpm * self.radius * math.pi / 30.0
        self.disk_area = math.pi * self.radius**2
        self.n_segments = int((1.0 - self.root_cutout) * 100)


@dataclass
class BladeGeometry:

    r_stations: List[float]
    chord_ft:   List[float]
    twist_deg:  List[float]

    def __post_init__(self) -> None:
        n = len(self.r_stations)
        if len(self.chord_ft) != n or len(self.twist_deg) != n:
            raise ValueError(
                "r_stations, chord_ft, and twist_deg must all have the same length"
            )
        if n < 2:
            raise ValueError("BladeGeometry requires at least 2 stations")
        r = np.asarray(self.r_stations)
        if not np.all(np.diff(r) > 0):
            raise ValueError("r_stations must be strictly increasing")

    def __len__(self) -> int:
        return len(self.r_stations)


@dataclass
class AirfoilZoneConfig:

    airfoil_ids: List[str] = field(default_factory=lambda: ["SC1095_CFD", "SC1095_CFD"])
    r_boundaries: List[float] = field(default_factory=lambda: [0.19, 1.0])

    def __post_init__(self) -> None:
        if len(self.airfoil_ids) != len(self.r_boundaries):
            raise ValueError(
                "airfoil_ids and r_boundaries must have the same length"
            )


@dataclass
class WeightSweepConfig:
    """Target rotor weight sweep (one BEMT solve per entry)."""

    target_weights_lb: List[float] = field(default_factory=lambda: [2200.0])
