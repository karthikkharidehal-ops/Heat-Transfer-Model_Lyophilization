"""
Pikal primary-drying model for tapered PCR-tube geometry.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, least_squares
from geometry import PCRTubeGeometry

DH_S = 2.838e6      # J/kg, latent heat of ice sublimation
RHO_ICE = 918.0     # kg/m^3


def p_ice(Tp_k: float) -> float:
    """Ice vapor pressure (Pa) vs temperature (K); valid ~-80C to 0C."""
    T = float(Tp_k)
    return float(np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T))


def solve_Tp(
    Ts_k: float, Pc_pa: float, Rp: float, Rs: float,
    Av_base: float, A_front: float, Kv: float
) -> tuple[float, float]:
    """
    Solve the coupled heat/mass balance for one timestep.
    Returns (mass flux per unit area, Tp in K).
    """
    def residual(Tp_k: float) -> float:
        flux = (p_ice(Tp_k) - Pc_pa) / (Rp + Rs) if (Rp + Rs) > 0 else 0.0
        flux = max(flux, 0.0)
        dmdt_total = flux * A_front
        q_supplied = Kv * (Ts_k - Tp_k) * Av_base
        q_consumed = dmdt_total * DH_S
        return float(q_supplied - q_consumed)

    lo, hi = Ts_k - 60.0, Ts_k - 0.01
    
    try:
        # Cast to float and ignore Pylance warning about brentq potentially returning a tuple
        Tp_k = float(brentq(residual, lo, hi, xtol=1e-4))  # type: ignore[arg-type]
    except ValueError:
        Tp_k = float(Ts_k - 5.0)
        
    flux = max((p_ice(Tp_k) - Pc_pa) / (Rp + Rs), 0.0) if (Rp + Rs) > 0 else 0.0
    return float(flux), float(Tp_k)


def simulate_cycle(
    t_s: np.ndarray, Ts_k: np.ndarray, Pc_pa: np.ndarray,
    tube: PCRTubeGeometry, fill_volume_m3: float,
    Kv: float, Rp0: float, A1: float, A2: float,
    Rs: float = 0.0
) -> np.ndarray:
    """Forward-simulate product temperature over a primary-drying window."""
    n = len(t_s)
    Tp_sim = np.zeros(n)
    Av_base = tube.base_area_m2()
    fill_height = tube.fill_height(fill_volume_m3)
    L = 0.0  # dried-layer thickness from the top, meters

    for i in range(n):
        Rp = Rp0 + A1 * L / (1.0 + A2 * L)
        A_front = tube.front_area_m2(fill_height, L)
        flux, Tp_k = solve_Tp(float(Ts_k[i]), float(Pc_pa[i]), float(Rp), float(Rs), float(Av_base), float(A_front), float(Kv))
        Tp_sim[i] = Tp_k

        if i < n - 1:
            dt = float(t_s[i + 1] - t_s[i])
            L += max(flux / RHO_ICE, 0.0) * dt
            L = min(L, fill_height)  # front can't pass the tube bottom

    return Tp_sim


def fit_parameters(
    t_s: np.ndarray, Ts_k: np.ndarray, Pc_pa: np.ndarray,
    Tp_measured_k: np.ndarray, tube: PCRTubeGeometry,
    fill_volume_m3: float, initial_guess: dict | None = None
) -> dict:
    """Fit {Kv, Rp0, A1, A2} to a measured Tp(t) trajectory."""
    guess = initial_guess or dict(Kv=15.0, Rp0=2e4, A1=1e6, A2=100.0)
    x0 = np.array([guess["Kv"], guess["Rp0"], guess["A1"], guess["A2"]], dtype=float)

    def resid(x: np.ndarray) -> np.ndarray:
        Kv, Rp0, A1, A2 = x
        Tp_sim = simulate_cycle(t_s, Ts_k, Pc_pa, tube, fill_volume_m3, Kv, Rp0, A1, A2)
        return Tp_sim - Tp_measured_k

    result = least_squares(resid, x0, bounds=(0, np.inf), xtol=1e-8, ftol=1e-8, max_nfev=2000)
    
    # Bypass Pylance's incomplete SciPy OptimizeResult stubs using getattr
    x_vals = getattr(result, "x", x0)
    fun_vals = getattr(result, "fun", np.zeros_like(x0))
    success = getattr(result, "success", False)
    
    fitted = dict(zip(["Kv", "Rp0", "A1", "A2"], x_vals))
    fitted["rms_error_k"] = float(np.sqrt(np.mean(fun_vals ** 2)))
    fitted["converged"] = bool(success)
    return fitted


@dataclass
class DryingSegment:
    """One contiguous primary-drying segment."""
    t_s: np.ndarray
    Ts_k: np.ndarray
    Pc_pa: np.ndarray
    Tp_measured_k: np.ndarray
    tube: PCRTubeGeometry
    fill_volume_m3: float
    label: str = ""


def fit_parameters_joint(
    segments: list[DryingSegment], 
    initial_guess: dict | None = None
) -> dict:
    """Fit ONE shared {Kv, Rp0, A1, A2} across multiple segments."""
    guess = initial_guess or dict(Kv=15.0, Rp0=2e4, A1=1e6, A2=100.0)
    x0 = np.array([guess["Kv"], guess["Rp0"], guess["A1"], guess["A2"]], dtype=float)

    def resid(x: np.ndarray) -> np.ndarray:
        Kv, Rp0, A1, A2 = x
        out = []
        for seg in segments:
            Tp_sim = simulate_cycle(
                seg.t_s, seg.Ts_k, seg.Pc_pa, seg.tube,
                seg.fill_volume_m3, Kv, Rp0, A1, A2
            )
            out.append(Tp_sim - seg.Tp_measured_k)
        return np.concatenate(out)

    result = least_squares(resid, x0, bounds=(0, np.inf), xtol=1e-8, ftol=1e-8, max_nfev=4000)
    
    # Bypass Pylance's incomplete SciPy OptimizeResult stubs
    x_vals = getattr(result, "x", x0)
    fun_vals = getattr(result, "fun", np.zeros_like(x0))
    success = getattr(result, "success", False)

    fitted = dict(zip(["Kv", "Rp0", "A1", "A2"], x_vals))
    fitted["rms_error_k"] = float(np.sqrt(np.mean(fun_vals ** 2)))
    fitted["converged"] = bool(success)

    per_segment = {}
    Kv, Rp0, A1, A2 = x_vals
    for seg in segments:
        Tp_sim = simulate_cycle(
            seg.t_s, seg.Ts_k, seg.Pc_pa, seg.tube,
            seg.fill_volume_m3, Kv, Rp0, A1, A2
        )
        per_segment[seg.label or f"segment_{id(seg)}"] = float(
            np.sqrt(np.mean((Tp_sim - seg.Tp_measured_k) ** 2))
        )
        
    fitted["per_segment_rms_k"] = per_segment
    return fitted

if __name__ == "__main__":
    print(__doc__)