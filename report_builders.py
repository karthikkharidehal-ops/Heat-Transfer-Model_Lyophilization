"""Shared report builders for mass-balance closure reporting.

Single source of truth for the five required mass-balance lines printed by
EVERY report builder (CLI report text, GUI log, and the saved fit_report.txt):

    Detected endpoint time (s): <value>
    Simulated endpoint time (s): <value>
    Mass-balance residual: <value>
    Mass-balance active: <True/False>
    Mass-balance penalty share of cost: <percent>

Event: E-FIX-MASSBAL-WIRE-001, E-FIX-MASSBAL-WIRE-002
"""

from typing import Optional


def _fmt(value: Optional[float]) -> str:
    """Format a float value, or 'None' when unavailable."""
    if value is None:
        return "None"
    return f"{float(value):.6f}"


def _fmt_percent(value: Optional[float]) -> str:
    """Format a fraction (0..1) as a percentage string, or 'None' when unavailable."""
    if value is None:
        return "None"
    return f"{float(value) * 100:.4f}%"


def build_mass_balance_lines(fitted: dict, detected_endpoint_time_s: Optional[float] = None) -> list[str]:
    """Build the exact five mass-balance closure lines from a fitted result dict.

    Parameters
    ----------
    fitted : dict
        Result of ``pikal_model.fit_parameters_joint``. Expected keys:
        ``simulated_endpoint_time_s``, ``mass_balance_residual``,
        ``mass_balance_active``, ``mass_balance_penalty_share``, and
        optionally ``detected_endpoint_time_s``.
    detected_endpoint_time_s : float, optional
        Fallback detected endpoint time (seconds from primary-drying start)
        used only if the fitted dict does not carry one.

    Returns
    -------
    list[str]
        Exactly five lines in the mandated format.
    """
    detected = fitted.get("detected_endpoint_time_s")
    if detected is None:
        detected = detected_endpoint_time_s

    simulated = fitted.get("simulated_endpoint_time_s")
    residual = fitted.get("mass_balance_residual")
    active = bool(fitted.get("mass_balance_active", False))
    penalty_share = fitted.get("mass_balance_penalty_share")

    return [
        f"Detected endpoint time (s): {_fmt(detected)}",
        f"Simulated endpoint time (s): {_fmt(simulated)}",
        f"Mass-balance residual: {_fmt(residual)}",
        f"Mass-balance active: {active}",
        f"Mass-balance penalty share of cost: {_fmt_percent(penalty_share)}",
    ]


def build_mass_balance_section(fitted: dict, detected_endpoint_time_s: Optional[float] = None) -> str:
    """Full '=== Mass-Balance Constraint ===' section as a single string block."""
    lines = ["=== Mass-Balance Constraint ==="]
    lines.extend(build_mass_balance_lines(fitted, detected_endpoint_time_s))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anchored-geometry reporting (E-GEOM-ANCHOR-001 / E-PHYS-GEOM-ANCHOR-001)
# ---------------------------------------------------------------------------

def build_geometry_lines(tube, profile: str) -> list[str]:
    """Active area-profile line plus the h vs V(h) vs A(h) table.

    Table runs from 0 to H in 0.5 mm steps. Single source of truth for
    BOTH GUIs and the CLI report (no duplicated report logic).
    """
    from geometry import build_geometry_table_rows

    lines = [
        "=== Anchored Geometry (E-GEOM-ANCHOR-001) ===",
        f"Geometry profile: {profile}",
        "Measured anchor: 16.0 uL = 4.0 mm fill height (fill height fixed at 4.0 mm; "
        "multi-point calibration DEFERRED - mandatory before recipe optimization or advisory MPC)",
        "",
        "h vs V(h) vs A(h) (0 to H in 0.5 mm steps):",
        f"  {'h (mm)':>8} | {'V(h) (uL)':>12} | {'A(h) (mm^2)':>12}",
        f"  {'-' * 8}-+-{'-' * 12}-+-{'-' * 12}",
    ]
    for h_m, v_m3, a_m2 in build_geometry_table_rows(tube, h_step_m=0.5e-3):
        lines.append(f"  {h_m * 1e3:8.2f} | {v_m3 * 1e9:12.6f} | {a_m2 * 1e6:12.6f}")
    return lines


def build_residual_diagnostics_lines(fitted: dict) -> list[str]:
    """Per-segment residual sign (simulated minus measured) statistics and
    median probe spread (max minus min across TP01..TP04 probes).

    Reads ``fitted['per_segment_diagnostics']`` produced by
    ``pikal_model.fit_parameters_joint``. Shared by both GUIs and the CLI.
    """
    diags = fitted.get("per_segment_diagnostics") or {}
    if not diags:
        return []
    lines = ["", "=== Per-Segment Residual Diagnostics (simulated - measured) ==="]
    for label, d in diags.items():
        mean_sign = _sign_word(d.get("mean_residual_k"))
        median_sign = _sign_word(d.get("median_residual_k"))
        lines.append(f"  {label}:")
        if d.get("mean_residual_k") is not None:
            lines.append(f"    Mean residual: {d['mean_residual_k']:+.6f} K ({mean_sign})")
        if d.get("median_residual_k") is not None:
            lines.append(f"    Median residual: {d['median_residual_k']:+.6f} K ({median_sign})")
        if d.get("residual_sign") is not None:
            lines.append(f"    Residual sign (median): {d['residual_sign']}")
        if d.get("median_probe_spread_k") is not None:
            lines.append(
                f"    Median probe spread (max-min of TP01..TP04): "
                f"{d['median_probe_spread_k']:.6f} K"
            )
        else:
            lines.append("    Median probe spread (max-min of TP01..TP04): unavailable (probe columns missing)")
    return lines


def _sign_word(value: Optional[float]) -> str:
    """Return 'positive' / 'negative' / 'zero-ish' for a residual statistic."""
    if value is None:
        return "unknown"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"
