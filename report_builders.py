"""Shared report builders for mass-balance closure reporting.

Single source of truth for the four required mass-balance lines printed by
EVERY report builder (CLI report text, GUI log, and the saved fit_report.txt):

    Detected endpoint time (s): <value>
    Simulated endpoint time (s): <value>
    Mass-balance residual: <value>
    Mass-balance active: <True/False>

Event: E-FIX-MASSBAL-WIRE-001
"""

from typing import Optional


def _fmt(value: Optional[float]) -> str:
    """Format a float value, or 'None' when unavailable."""
    if value is None:
        return "None"
    return f"{float(value):.6f}"


def build_mass_balance_lines(fitted: dict, detected_endpoint_time_s: Optional[float] = None) -> list[str]:
    """Build the exact four mass-balance closure lines from a fitted result dict.

    Parameters
    ----------
    fitted : dict
        Result of ``pikal_model.fit_parameters_joint``. Expected keys:
        ``simulated_endpoint_time_s``, ``mass_balance_residual``,
        ``mass_balance_active``, and optionally ``detected_endpoint_time_s``.
    detected_endpoint_time_s : float, optional
        Fallback detected endpoint time (seconds from primary-drying start)
        used only if the fitted dict does not carry one.

    Returns
    -------
    list[str]
        Exactly four lines in the mandated format.
    """
    detected = fitted.get("detected_endpoint_time_s")
    if detected is None:
        detected = detected_endpoint_time_s

    simulated = fitted.get("simulated_endpoint_time_s")
    residual = fitted.get("mass_balance_residual")
    active = bool(fitted.get("mass_balance_active", False))

    return [
        f"Detected endpoint time (s): {_fmt(detected)}",
        f"Simulated endpoint time (s): {_fmt(simulated)}",
        f"Mass-balance residual: {_fmt(residual)}",
        f"Mass-balance active: {active}",
    ]


def build_mass_balance_section(fitted: dict, detected_endpoint_time_s: Optional[float] = None) -> str:
    """Full '=== Mass-Balance Constraint ===' section as a single string block."""
    lines = ["=== Mass-Balance Constraint ==="]
    lines.extend(build_mass_balance_lines(fitted, detected_endpoint_time_s))
    return "\n".join(lines)
