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
    Av_base: float, A_front: float, Kv: float,
    Tp_prev_k: float | None = None, dt: float | None = None,
    ice_mass: float | None = None, cp_ice: float = 2090.0
) -> tuple[float, float]:
    """
    Solve the coupled heat/mass balance for one timestep.
    
    Parameters
    ----------
    Ts_k : float
        Shelf temperature (K)
    Pc_pa : float
        Chamber pressure (Pa)
    Rp : float
        Product resistance (m²·s·Pa/kg)
    Rs : float
        Stopper/chamber resistance (m²·s·Pa/kg)
    Av_base : float
        Vial base area (m²)
    A_front : float
        Sublimation front area (m²)
    Kv : float
        Heat transfer coefficient (W/m²·K)
    Tp_prev_k : float, optional
        Product temperature at previous timestep (K). Required for transient mode.
    dt : float, optional
        Timestep duration (s). Required for transient mode.
    ice_mass : float, optional
        Remaining ice mass (kg). Required for transient mode.
    cp_ice : float, default 2090.0
        Specific heat capacity of ice (J/kg·K)
    
    Returns
    -------
    flux : float
        Mass flux per unit area (kg/m²·s)
    Tp_k : float
        Product temperature at current timestep (K)
    
    Notes
    -----
    When Tp_prev_k, dt, and ice_mass are provided, the solver includes a transient
    heat accumulation term to account for non-steady conditions during ramp steps:
    
        q_supplied = q_consumed + q_accumulated
    
    where q_accumulated = (ice_mass * cp_ice * (Tp_k - Tp_prev_k)) / dt
    
    This allows the model to handle multi-step ramps where the steady-state 
    assumption breaks down.
    """
    def residual(Tp_k: float) -> float:
        flux = (p_ice(Tp_k) - Pc_pa) / (Rp + Rs) if (Rp + Rs) > 0 else 0.0
        flux = max(flux, 0.0)
        dmdt_total = flux * A_front
        q_supplied = Kv * (Ts_k - Tp_k) * Av_base
        q_consumed = dmdt_total * DH_S
        
        # Add transient heat accumulation term if parameters provided
        if Tp_prev_k is not None and dt is not None and dt > 0 and ice_mass is not None:
            q_accumulated = (ice_mass * cp_ice * (Tp_k - Tp_prev_k)) / dt
            return float(q_supplied - q_consumed - q_accumulated)
        else:
            # Steady-state mode (original behavior)
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
    Rs: float = 0.0, use_transient: bool = False
) -> np.ndarray:
    """Forward-simulate product temperature over a primary-drying window.
    
    Parameters
    ----------
    t_s : np.ndarray
        Time array (seconds)
    Ts_k : np.ndarray
        Shelf temperature history (K)
    Pc_pa : np.ndarray
        Chamber pressure history (Pa)
    tube : PCRTubeGeometry
        Geometry object for the PCR tube
    fill_volume_m3 : float
        Fill volume in cubic meters
    Kv : float
        Heat transfer coefficient (W/m²·K)
    Rp0, A1, A2 : float
        Product resistance parameters
    Rs : float, default 0.0
        Stopper/chamber resistance (m²·s·Pa/kg)
    use_transient : bool, default False
        If True, include transient heat accumulation term to account for
        non-steady conditions during ramp steps. Recommended when there are
        4-5 ramp steps where steady-state assumption breaks down.
        
    Returns
    -------
    Tp_sim : np.ndarray
        Simulated product temperature history (K)
    """
    n = len(t_s)
    Tp_sim = np.zeros(n)
    Av_base = tube.base_area_m2()
    fill_height = tube.fill_height(fill_volume_m3)
    L = 0.0  # dried-layer thickness from the top, meters
    
    # Initialize for transient mode
    if use_transient:
        initial_ice_volume = fill_volume_m3  # Start with full ice volume
        initial_ice_mass = RHO_ICE * initial_ice_volume
        ice_mass = initial_ice_mass
        Tp_prev = Ts_k[0] - 5.0  # Initial guess for product temp
    else:
        ice_mass = None
        Tp_prev = None
    
    for i in range(n):
        Rp = Rp0 + A1 * L / (1.0 + A2 * L)
        A_front = tube.front_area_m2(fill_height, L)
        
        if use_transient and i > 0:
            dt = float(t_s[i] - t_s[i - 1])
            flux, Tp_k = solve_Tp(
                float(Ts_k[i]), float(Pc_pa[i]), float(Rp), float(Rs),
                float(Av_base), float(A_front), float(Kv),
                Tp_prev_k=float(Tp_prev), dt=dt, ice_mass=float(ice_mass)
            )
            # Update ice mass based on sublimation
            dmdt_total = flux * A_front
            ice_mass -= max(dmdt_total * dt, 0.0)
            ice_mass = max(ice_mass, 0.0)  # Can't go negative
            Tp_prev = Tp_k
        else:
            flux, Tp_k = solve_Tp(
                float(Ts_k[i]), float(Pc_pa[i]), float(Rp), float(Rs),
                float(Av_base), float(A_front), float(Kv)
            )
        
        Tp_sim[i] = Tp_k

        if i < n - 1:
            dt = float(t_s[i + 1] - t_s[i])
            L += max(flux / RHO_ICE, 0.0) * dt
            L = min(L, fill_height)  # front can't pass the tube bottom

    return Tp_sim


def fit_parameters(
    t_s: np.ndarray, Ts_k: np.ndarray, Pc_pa: np.ndarray,
    Tp_measured_k: np.ndarray, tube: PCRTubeGeometry,
    fill_volume_m3: float, initial_guess: dict | None = None,
    use_transient: bool = False
) -> dict:
    """Fit {Kv, Rp0, A1, A2} to a measured Tp(t) trajectory.
    
    Parameters
    ----------
    t_s : np.ndarray
        Time array (seconds)
    Ts_k : np.ndarray
        Shelf temperature history (K)
    Pc_pa : np.ndarray
        Chamber pressure history (Pa)
    Tp_measured_k : np.ndarray
        Measured product temperature history (K)
    tube : PCRTubeGeometry
        Geometry object
    fill_volume_m3 : float
        Fill volume in cubic meters
    initial_guess : dict, optional
        Initial guess for parameters
    use_transient : bool, default False
        If True, use transient mode for simulation (recommended for multi-step ramps)
    """
    guess = initial_guess or dict(Kv=15.0, Rp0=2e4, A1=1e6, A2=100.0)
    x0 = np.array([guess["Kv"], guess["Rp0"], guess["A1"], guess["A2"]], dtype=float)

    def resid(x: np.ndarray) -> np.ndarray:
        Kv, Rp0, A1, A2 = x
        Tp_sim = simulate_cycle(t_s, Ts_k, Pc_pa, tube, fill_volume_m3, Kv, Rp0, A1, A2, use_transient=use_transient)
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
    is_hold_step: bool = True  # Default to hold step (steady-state)


def fit_parameters_joint(
    segments: list[DryingSegment], 
    initial_guess: dict | None = None,
    use_transient: bool = False,
    use_hybrid: bool = False
) -> dict:
    """Fit ONE shared {Kv, Rp0, A1, A2} across multiple segments.
    
    Parameters
    ----------
    segments : list[DryingSegment]
        List of drying segments to fit jointly
    initial_guess : dict, optional
        Initial guess for parameters
    use_transient : bool, default False
        If True, use transient mode for simulation (recommended for multi-step ramps
        where steady-state assumption breaks down)
    use_hybrid : bool, default False
        If True, apply quasi-steady state for Hold steps (detected by shelf_setpt stability)
        and transient mode for Ramp steps. This overrides use_transient when enabled.
    """
    guess = initial_guess or dict(Kv=15.0, Rp0=2e4, A1=1e6, A2=100.0)
    x0 = np.array([guess["Kv"], guess["Rp0"], guess["A1"], guess["A2"]], dtype=float)

    def resid(x: np.ndarray) -> np.ndarray:
        Kv, Rp0, A1, A2 = x
        out = []
        for seg in segments:
            # Determine if this segment should use transient or steady-state
            seg_use_transient = use_transient
            if use_hybrid:
                # Use the is_hold_step flag from the segment
                # Hold steps -> steady-state (False), Ramp steps -> transient (True)
                seg_use_transient = not seg.is_hold_step
            
            Tp_sim = simulate_cycle(
                seg.t_s, seg.Ts_k, seg.Pc_pa, seg.tube,
                seg.fill_volume_m3, Kv, Rp0, A1, A2, use_transient=seg_use_transient
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
        # Determine if this segment should use transient or steady-state for reporting
        seg_use_transient = use_transient
        if use_hybrid:
            seg_use_transient = not seg.is_hold_step
        
        Tp_sim = simulate_cycle(
            seg.t_s, seg.Ts_k, seg.Pc_pa, seg.tube,
            seg.fill_volume_m3, Kv, Rp0, A1, A2, use_transient=seg_use_transient
        )
        per_segment[seg.label or f"segment_{id(seg)}"] = float(
            np.sqrt(np.mean((Tp_sim - seg.Tp_measured_k) ** 2))
        )
        
    fitted["per_segment_rms_k"] = per_segment
    return fitted

if __name__ == "__main__":
    print(__doc__)