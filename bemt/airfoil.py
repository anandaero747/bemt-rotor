"""Airfoil coefficient tables: parsing, storage, database, and resolver."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

# Default directory for bundled PRASADUM-format airfoil tables.
_DATA_DIR = Path(__file__).parent / "data"

# Bundled airfoil names (filename stems in bemt/data/).
BUNDLED_AIRFOILS = [
    "SC1095_CFD",
    "SC1095_base",
    "SC1095_TNN",
    "SC1095_Opt_TNN_Mach",
    "SC1095_CFD_pt_Baseline",
    "SC1095_CFD_pt_Case1",
    "SC1095_CFD_pt_Case2",
    "SC1095_NN_pt_Baseline",
    "SC1095_NN_pt_Case1",
    "SC1095_NN_pt_Case2",
    "SC1095_Pt_Case2",
    "SC1095_CFD_Case2",
]


@dataclass
class AirfoilTable:
    """Cl, Cd, Cm tables for one airfoil indexed by Mach and angle of attack.

    Data file format (PRASADUM-style):
        Line 1  : description header (skipped)
        Line 2  : "nMach nAoA ..."
        Lines 3–5 : separator / "-- Lift coefficient" / separator
        nMach × nAoA rows : "Mach  AoA_deg  Cl"
        (same pattern for Cd, then Cm)
    """

    mach_vals: np.ndarray   # shape (n_mach,)
    alpha_vals: np.ndarray  # shape (n_alpha,), degrees
    cl: np.ndarray          # shape (n_mach, n_alpha)
    cd: np.ndarray          # shape (n_mach, n_alpha)
    cm: np.ndarray          # shape (n_mach, n_alpha)

    @property
    def n_mach(self) -> int:
        return len(self.mach_vals)

    @property
    def n_alpha(self) -> int:
        return len(self.alpha_vals)

    @classmethod
    def from_file(cls, filepath: Path) -> AirfoilTable:
        """Parse a PRASADUM-style airfoil coefficient file."""
        with open(filepath, "r") as fid:
            fid.readline()                          # skip description
            header = fid.readline().split()
            n_mach, n_alpha = int(header[0]), int(header[1])

            def _read_section() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                for _ in range(3):                  # separator / label / separator
                    fid.readline()
                mach_col = np.zeros(n_mach)
                alpha_col = np.zeros(n_alpha)
                data = np.zeros((n_mach, n_alpha))
                for j in range(n_mach):
                    for k in range(n_alpha):
                        row = list(map(float, fid.readline().split()))
                        mach_col[j] = row[0]
                        alpha_col[k] = row[1]
                        data[j, k] = row[2]
                return mach_col, alpha_col, data

            mach_vals, alpha_vals, cl = _read_section()
            _, _, cd = _read_section()
            _, _, cm = _read_section()

        return cls(mach_vals=mach_vals, alpha_vals=alpha_vals, cl=cl, cd=cd, cm=cm)

    @classmethod
    def from_c81(cls, filepath: Path) -> AirfoilTable:
        """Parse a C81-format airfoil coefficient file.

        Expected layout::

            C81 Table

            Cl vs alpha & Mach
            alpha\\tM0.1\\tM0.2\\tM0.3  ...
            -180.00  0.04   0.04   0.04  ...
            ...

            Cd vs alpha & Mach
            ...

            Cm vs alpha & Mach
            ...

        Rows are AoA values; columns are Mach numbers extracted from the
        header (e.g. "M0.3" → 0.3).  All three sections must be present.
        """
        with open(filepath, "r") as f:
            lines = f.readlines()

        section_lines: dict[str, int] = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("Cl vs alpha"):
                section_lines["cl"] = i
            elif stripped.startswith("Cd vs alpha"):
                section_lines["cd"] = i
            elif stripped.startswith("Cm vs alpha"):
                section_lines["cm"] = i

        if len(section_lines) != 3:
            raise ValueError(
                f"{filepath}: could not find all three C81 sections "
                f"(Cl / Cd / Cm). Found: {list(section_lines)}"
            )

        def _parse_section(start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            col_header = lines[start + 1].strip().split("\t")
            mach_vals = np.array([float(h.lstrip("M")) for h in col_header[1:]])

            alpha_list, data_rows = [], []
            for line in lines[start + 2:]:
                stripped = line.strip()
                if not stripped:
                    break
                parts = stripped.split()
                alpha_list.append(float(parts[0]))
                data_rows.append([float(v) for v in parts[1:]])

            alpha_vals = np.array(alpha_list)
            data = np.array(data_rows)    # shape (n_alpha, n_mach)
            return mach_vals, alpha_vals, data.T  # → (n_mach, n_alpha)

        mach_vals, alpha_vals, cl = _parse_section(section_lines["cl"])
        _,         _,           cd = _parse_section(section_lines["cd"])
        _,         _,           cm = _parse_section(section_lines["cm"])

        return cls(mach_vals=mach_vals, alpha_vals=alpha_vals, cl=cl, cd=cd, cm=cm)


class AirfoilDatabase:
    """Loads and caches AirfoilTable objects from a directory by name.

    Parameters
    ----------
    airfoil_dir : directory containing PRASADUM-format .dat files.
                  Defaults to the bundled ``bemt/data/`` directory.
    """

    def __init__(self, airfoil_dir: Path = _DATA_DIR) -> None:
        self._dir = Path(airfoil_dir)
        self._cache: Dict[str, AirfoilTable] = {}

    def get(self, name: str) -> AirfoilTable:
        """Return the AirfoilTable for ``name`` (filename stem, no .dat)."""
        if name not in self._cache:
            self._cache[name] = AirfoilTable.from_file(self._dir / f"{name}.dat")
        return self._cache[name]

    def load_names(self, names: list[str]) -> Dict[str, AirfoilTable]:
        """Pre-load all requested names and return a mapping."""
        for name in set(names):
            self.get(name)
        return {name: self._cache[name] for name in set(names)}


class AirfoilResolver:
    """Resolves airfoil names to AirfoilTable objects.

    Each airfoil is defined by a dict with a ``source`` key:

    .. code-block:: yaml

        SC1095:
          source: agrc          # generate via AGRC surrogate
          geometry: SC1095.dat  # airfoil coordinate file

        SC1094R8:
          source: c81           # load pre-computed C81 from c81_dir/
          # file: override.dat  # optional: explicit path instead of c81_dir

        SC1095_CFD:
          source: builtin       # bundled PRASADUM table in bemt/data/

    Parameters
    ----------
    definitions : mapping of airfoil name → definition dict
    c81_dir     : directory searched for ``{name}.dat`` when source is "c81"
    agrc_dir    : path to ``agrc_surrogate/`` when source is "agrc"
    """

    def __init__(
        self,
        definitions: dict[str, dict],
        c81_dir: str | Path | None = None,
        agrc_dir: str | Path | None = None,
    ) -> None:
        self._defs = definitions
        self._c81_dir = Path(c81_dir) if c81_dir else None
        self._agrc_dir = Path(agrc_dir).expanduser() if agrc_dir else None
        self._cache: dict[str, AirfoilTable] = {}
        self._agrc = None   # lazy-loaded on first AGRC request

    def resolve(self, name: str) -> AirfoilTable:
        """Return the AirfoilTable for ``name``, loading if necessary."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def resolve_zone(self, airfoil_ids: list[str]) -> list[AirfoilTable]:
        """Return one AirfoilTable per zone boundary (same order as airfoil_ids)."""
        return [self.resolve(name) for name in airfoil_ids]

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load(self, name: str) -> AirfoilTable:
        if name not in self._defs:
            raise KeyError(
                f"Airfoil '{name}' not found in definitions. "
                f"Available: {list(self._defs)}"
            )
        defn = self._defs[name]
        source = defn.get("source", "builtin")

        if source == "builtin":
            return AirfoilTable.from_file(_DATA_DIR / f"{name}.dat")

        if source == "c81":
            path = Path(defn["file"]) if "file" in defn else self._c81_path(name)
            return AirfoilTable.from_c81(path)

        if source == "agrc":
            return self._from_agrc(name, defn)

        raise ValueError(
            f"Unknown source '{source}' for airfoil '{name}'. "
            f"Valid options: builtin, c81, agrc"
        )

    def _c81_path(self, name: str) -> Path:
        if self._c81_dir is None:
            raise ValueError(
                f"Airfoil '{name}' has source 'c81' but no c81_dir was specified "
                f"and no explicit 'file' key was given."
            )
        candidates = [
            self._c81_dir / f"{name}.dat",
            self._c81_dir / f"{name}_C81.dat",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"C81 file for '{name}' not found. Searched:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )

    def _from_agrc(self, name: str, defn: dict) -> AirfoilTable:
        agrc_dir = self._agrc_dir
        if "agrc_dir" in defn:
            agrc_dir = Path(defn["agrc_dir"]).expanduser()
        if agrc_dir is None:
            raise ValueError(
                f"Airfoil '{name}' has source 'agrc' but no agrc_dir was specified."
            )

        if self._agrc is None:
            try:
                from pipeline import load_agrc_models
            except ImportError as exc:
                raise ImportError(
                    "pipeline.py dependencies not installed. "
                    "Run: pip install bemt-rotor[agrc]"
                ) from exc
            self._agrc = load_agrc_models(agrc_dir)

        from pipeline import split_surfaces, fit_cst, predict_c81

        geom = Path(defn["geometry"])
        if not geom.is_absolute() and "geometry_base" in defn:
            geom = Path(defn["geometry_base"]) / geom

        xU, yU, xL, yL = split_surfaces(geom)
        xU, yU = xU[np.argsort(xU)], yU[np.argsort(xU)]
        xL, yL = xL[np.argsort(xL)], yL[np.argsort(xL)]
        X_input  = fit_cst(xU, yU, xL, yL)
        X_scaled = self._agrc.scaler_x.transform(X_input)

        print(f"  [{name}] Running AGRC surrogate…")
        return predict_c81(self._agrc, X_scaled)
