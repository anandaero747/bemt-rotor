"""YAML configuration loader for BEMT run inputs."""
from __future__ import annotations

from pathlib import Path

from .config import (
    AtmosphericConfig,
    AirfoilZoneConfig,
    BladeGeometry,
    RotorConfig,
    WeightSweepConfig,
)


def load_config(path: str | Path) -> tuple[
    AtmosphericConfig,
    RotorConfig,
    BladeGeometry,
    AirfoilZoneConfig,
    WeightSweepConfig,
    dict,      # airfoil definitions (name → {source, ...})
    Path | None,  # c81_dir
    Path | None,  # agrc_dir
]:
    """Load a BEMT run configuration from a YAML file.

    All relative paths inside the YAML are resolved relative to the
    directory that contains the YAML file itself.

    Returns
    -------
    atmo, rotor, blade, airfoil_zone, sweep, airfoil_defs, c81_dir, agrc_dir
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required to load YAML config files. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(path).expanduser().resolve()
    base = path.parent

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # ── Rotor ────────────────────────────────────────────────────────────────
    r = cfg["rotor"]
    rotor = RotorConfig(
        radius     = r["radius_ft"],
        n_blades   = r["n_blades"],
        n_rotors   = r.get("n_rotors", 1),
        root_cutout= r["root_cutout"],
        tip_speed  = r["tip_speed_fps"],
        tip_loss   = r.get("tip_loss", True),
    )

    # ── Atmosphere ───────────────────────────────────────────────────────────
    a = cfg.get("atmosphere", {})
    atmo = AtmosphericConfig(
        altitude = a.get("altitude_ft", 0.0),
        T_today  = a.get("T_today_F",  59.0),
    )

    # ── Blade geometry ───────────────────────────────────────────────────────
    b = cfg["blade"]
    blade = BladeGeometry(
        r_stations = b["r_stations"],
        chord_ft   = b["chord_ft"],
        twist_deg  = b["twist_deg"],
    )

    # ── Airfoil zones ─────────────────────────────────────────────────────────
    zones = b["airfoil_zones"]
    airfoil_zone = AirfoilZoneConfig(
        airfoil_ids  = [z["airfoil"] for z in zones],
        r_boundaries = [z["r"]       for z in zones],
    )

    # ── Sweep ────────────────────────────────────────────────────────────────
    s = cfg.get("sweep", {})
    sweep = WeightSweepConfig(
        target_weights_lb = s.get("weights_lb", [2200.0])
    )

    # ── Airfoil definitions (resolve geometry/file paths relative to YAML) ──
    airfoil_defs: dict[str, dict] = {}
    for name, defn in cfg.get("airfoils", {}).items():
        defn = dict(defn)
        if "file" in defn:
            defn["file"] = str((base / defn["file"]).resolve())
        if "geometry" in defn:
            # Attach base dir so AirfoilResolver can resolve relative geometry
            defn["geometry"] = str((base / defn["geometry"]).resolve())
        airfoil_defs[name] = defn

    # ── Global path settings ─────────────────────────────────────────────────
    c81_dir = None
    if "c81_dir" in cfg:
        c81_dir = (base / cfg["c81_dir"]).expanduser().resolve()

    agrc_dir = None
    if "agrc_dir" in cfg:
        agrc_dir = Path(cfg["agrc_dir"]).expanduser().resolve()

    return atmo, rotor, blade, airfoil_zone, sweep, airfoil_defs, c81_dir, agrc_dir
