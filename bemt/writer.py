"""Formatted text output writer for BEMT results."""
from __future__ import annotations

import datetime
from pathlib import Path

from .config import AtmosphericConfig, RotorConfig
from .results import BEMTResult

_SEP  = "=" * 72
_DASH = "-" * 72


def write_output(
    results: list[BEMTResult],
    weights_lb: list[float],
    rotor: RotorConfig,
    atmo: AtmosphericConfig,
    output_path: str | Path,
    input_file: str | Path | None = None,
) -> None:
    lines: list[str] = []

    def h(text: str = "") -> None:
        lines.append(text)

    # ── Header ────────────────────────────────────────────────────────────────
    h(_SEP)
    h("BEMT ROTOR ANALYSIS — OUTPUT")
    h(_SEP)
    h(f"Generated  : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    if input_file:
        h(f"Input file : {Path(input_file).name}")
    h()

    # ── Rotor parameters ──────────────────────────────────────────────────────
    h("ROTOR PARAMETERS")
    h(f"  Radius              : {rotor.radius:>8.2f} ft")
    if rotor.rpm is not None:
        h(f"  RPM                 : {rotor.rpm:>8.1f}")
    h(f"  Tip speed           : {rotor.tip_speed:>8.1f} ft/s")
    h(f"  Blades              : {rotor.n_blades:>8d}")
    if rotor.n_rotors > 1:
        h(f"  Rotors              : {rotor.n_rotors:>8d}")
    h(f"  Root cutout         : {rotor.root_cutout:>8.3f} r/R")
    h(f"  Tip-loss correction : {'ON' if rotor.tip_loss else 'OFF':>8s}")
    h()

    # ── Atmospheric conditions ─────────────────────────────────────────────────
    h("ATMOSPHERIC CONDITIONS")
    h(f"  Altitude            : {atmo.altitude:>8.0f} ft")
    h(f"  Temperature         : {atmo.T_today:>8.1f} °F")
    h(f"  Density             : {atmo.rho:>10.6f} slug/ft³")
    h(f"  Speed of sound      : {atmo.aspeed:>8.2f} ft/s")
    h()

    # ── Per-case results ───────────────────────────────────────────────────────
    for i, (w, res) in enumerate(zip(weights_lb, results), start=1):
        h(_SEP)
        h(f"CASE {i}  |  Target weight: {w:.0f} lb")
        h(_SEP)
        h()

        if res.bisection_fail:
            h("  WARNING: collective bisection did not converge — results may be unreliable.")
        if res.lambda_fail:
            h("  WARNING: inflow iteration did not converge — results may be unreliable.")

        h(f"  CT (total)          : {res.CT:.6f}")
        h(f"  CPi (induced)       : {res.CPi:.6f}")
        h(f"  CP0 (profile)       : {res.CP0:.6f}")
        h(f"  CP (total)          : {res.CPi + res.CP0:.6f}")
        h(f"  Figure of merit     : {res.FM:.4f}")
        h(f"  Collective pitch    : {res.collective_deg:.2f}°")
        h(f"  θ₇₅                 : {res.theta_75_deg:.2f}°")
        h(f"  Mean AoA            : {res.mean_alpha_deg:.2f}°")
        h(f"  Power loading       : {res.PL:.2f} lb/hp")
        h(f"  Torque              : {res.torque:.1f} lb·ft")
        h(f"  κ (induced factor)  : {res.kappa:.4f}")
        h()

        if res.r_span.size > 0:
            h("  SPANWISE DISTRIBUTION")
            h("  " + _DASH)
            h(f"  {'r/R':>6}  {'AoA(°)':>8}  {'Cl':>8}  {'Cd':>8}"
              f"  {'dCT':>10}  {'dCPi':>10}  {'dCP0':>10}")
            h("  " + _DASH)
            for j in range(len(res.r_span)):
                h(
                    f"  {res.r_span[j]:>6.3f}"
                    f"  {res.alpha_span_deg[j]:>8.3f}"
                    f"  {res.cl_span[j]:>8.4f}"
                    f"  {res.cd_span[j]:>8.5f}"
                    f"  {res.dCT_span[j]:>10.3e}"
                    f"  {res.dCPi_span[j]:>10.3e}"
                    f"  {res.dCP0_span[j]:>10.3e}"
                )
            h("  " + _DASH)
        h()

    h(_SEP)
    h("END OF OUTPUT")
    h(_SEP)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Output written to: {output_path}")
