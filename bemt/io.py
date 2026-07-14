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
    dict,           # airfoil definitions (name → {source, ...})
    Path | None,    # c81_dir
    Path | None,    # agrc_dir
    Path | None,    # output file path
]:
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
        radius      = r["radius_ft"],
        n_blades    = r["n_blades"],
        n_rotors    = r.get("n_rotors", 1),
        root_cutout = r.get("root_cutout", cfg["blade"]["r_stations"][0]),
        rpm         = r.get("rpm"),
        tip_speed   = r.get("tip_speed_fps", 550.0),
        tip_loss    = r.get("tip_loss", True),
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

    # ── Airfoil zones — two supported formats ─────────────────────────────────
    #
    #  Format A (airfoil_geometry): per-station coordinate files → AGRC surrogate
    #    airfoil_geometry:
    #      - r: 0.19
    #        file: airfoils/root.dat
    #      - r: 1.00
    #        file: airfoils/tip.dat
    #
    #  Format B (airfoil_zones): named airfoils resolved via the airfoils: block
    #    airfoil_zones:
    #      - r: 0.19
    #        airfoil: SC1095_CFD
    #
    airfoil_defs: dict[str, dict] = {}

    if "airfoil_geometry" in b:
        geom_list = b["airfoil_geometry"]
        airfoil_ids, r_boundaries = [], []
        for entry in geom_list:
            geom_path = (base / entry["file"]).resolve()
            name = geom_path.stem
            airfoil_ids.append(name)
            r_boundaries.append(float(entry["r"]))
            airfoil_defs[name] = {
                "source": "agrc",
                "geometry": str(geom_path),
            }
        airfoil_zone = AirfoilZoneConfig(
            airfoil_ids=airfoil_ids,
            r_boundaries=r_boundaries,
        )

    elif "airfoil_zones" in b:
        zones = b["airfoil_zones"]
        airfoil_zone = AirfoilZoneConfig(
            airfoil_ids  = [z["airfoil"] for z in zones],
            r_boundaries = [z["r"]       for z in zones],
        )

    else:
        raise KeyError(
            "blade section must contain either 'airfoil_geometry' "
            "(coordinate files for AGRC) or 'airfoil_zones' (named airfoils)."
        )

    # ── Sweep ────────────────────────────────────────────────────────────────
    s = cfg.get("sweep", {})
    sweep = WeightSweepConfig(
        target_weights_lb = s.get("weights_lb", [2200.0])
    )

    # ── Airfoil definitions from airfoils: block ──────────────────────────────
    for name, defn in cfg.get("airfoils", {}).items():
        defn = dict(defn)
        if "file" in defn:
            defn["file"] = str((base / defn["file"]).resolve())
        if "geometry" in defn:
            defn["geometry"] = str((base / defn["geometry"]).resolve())
        airfoil_defs[name] = defn

    # ── Global path settings ─────────────────────────────────────────────────
    c81_dir = None
    if "c81_dir" in cfg:
        c81_dir = (base / cfg["c81_dir"]).expanduser().resolve()

    agrc_dir = None
    if "agrc_dir" in cfg:
        agrc_dir = Path(cfg["agrc_dir"]).expanduser().resolve()

    # ── Output file ──────────────────────────────────────────────────────────
    output_path = None
    if "output" in cfg:
        output_path = (base / cfg["output"]).resolve()

    return atmo, rotor, blade, airfoil_zone, sweep, airfoil_defs, c81_dir, agrc_dir, output_path
