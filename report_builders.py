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
