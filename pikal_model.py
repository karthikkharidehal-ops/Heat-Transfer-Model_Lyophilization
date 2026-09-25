"""
Pikal primary-drying model for tapered PCR-tube geometry.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, least_squares
from geometry import PCRTubeGeometry, AnchoredTubeGeometry

DH_S = 2.838e6      # J/kg, latent heat of ice sublimation
RHO_ICE = 918.0     # kg/m^3


class NoRootError(RuntimeError):
    """Raised when the steady-state/transient heat/mass balance has NO root
    for Tp in [Ts - 60, Ts) (event E-PHYS-STATELAW-001, defect 3).

    Physically this means the supplied heat Kv*(Ts - Tp)*contact_area exceeds
    the maximum sublimation capacity (p_ice(Tp) - Pc)/(Rp + Rs) * A_front * DH_S
    over the entire admissible Tp window: there is NO product temperature at
    which the energy balance closes. This is a STRUCTURAL model failure (Kv or
    contact area too high), not a numerical hiccup — it must never be masked
    by silently pinning Tp := Ts - 5.0.

    Carries the full solver context so callers can exclude the timestep from
    residuals and report why no root exists.
    """

    def __init__(self, Ts, Pc, Rp, contact_area, A_front, message=None):
        self.Ts = float(Ts)
        self.Pc = float(Pc)
        self.Rp = float(Rp)
        self.contact_area = float(contact_area)
        self.A_front = float(A_front)
        if message is None:
            message = (
                "No root in heat/mass balance for Tp in [Ts-60, Ts): "
                f"Ts={self.Ts:.4f} K, Pc={self.Pc:.4f} Pa, Rp={self.Rp:.6g} "
                f"m*s*Pa/kg, contact_area={self.contact_area:.6g} m^2, "
                f"A_front={self.A_front:.6g} m^2. Heat input exceeds "
                "sublimation capacity (Kv or contact area structurally too high)."
            )
        super().__init__(message)


def p_ice(Tp_k: float) -> float:
    """Ice vapor pressure (Pa) vs temperature (K); valid ~-80C to 0C."""
    T = float(Tp_k)
    return float(np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T))


def solve_Tp(
    Ts_k: float, Pc_pa: float, Rp: float, Rs: float,
    contact_area: float, A_front: float, Kv: float,
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
    contact_area : float
        Heat transfer contact area (m²) - lateral surface area + base
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

    Raises
    ------
    NoRootError
        If the heat/mass balance has NO root for Tp in [Ts-60, Ts). The silent
        fallback Tp := Ts - 5.0 that previously masked this case is ABOLISHED
        (event E-PHYS-STATELAW-001, defect 3): it manufactured a constant
        ~-5.7 K Hold-step residual by pretending a physical impossibility was
        merely an awkward solve. Callers must catch NoRootError per timestep,
        exclude that timestep from residuals, and count exclusions.

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
        q_supplied = Kv * (Ts_k - Tp_k) * contact_area
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
    except ValueError as exc:
        # NO SILENT FALLBACK (E-PHYS-STATELAW-001, defect 3): brentq's
        # ValueError means residual(lo) and residual(hi) share the same sign,
        # i.e. there is NO admissible product temperature at which the energy
        # balance closes. The old code masked this by pinning Tp := Ts - 5.0,
        # manufacturing a constant Hold-step bias. Raise instead; callers must
        # catch per timestep, exclude it from residuals, and count exclusions.
        raise NoRootError(Ts_k, Pc_pa, Rp, contact_area, A_front) from exc

    flux = max((p_ice(Tp_k) - Pc_pa) / (Rp + Rs), 0.0) if (Rp + Rs) > 0 else 0.0
    return float(flux), float(Tp_k)


def simulate_cycle(
    t_s: np.ndarray, Ts_k: np.ndarray, Pc_pa: np.ndarray,
    tube: PCRTubeGeometry, fill_volume_m3: float,
    Kv: float, Rp0: float, A1: float, A2: float,
    Rs: float = 0.0, use_transient: bool = False, return_masks: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray, bool]:
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
    return_masks : bool, default False
        If True, return (Tp_sim, valid_mask, endpoint_reached) instead of
        Tp_sim only. valid_mask[i] is False for post-primary timesteps
        (at/after the simulated endpoint L >= H) and must be used to exclude
        them from residual computations.

    Returns
    -------
    Tp_sim : np.ndarray
        Simulated product temperature history (K). Timesteps at/after the
        simulated endpoint (first time L >= H) carry NaN and MUST be excluded
        from any residual computation (post-primary: the primary-drying model
        never simulates through an empty tube).
    valid_mask : np.ndarray (bool), endpoint_reached : bool
        Only returned when ``return_masks=True``.

    Raises
    ------
    NoRootError
        Propagated from solve_Tp when the heat/mass balance has no root
        (silent Tp := Ts - 5.0 fallback abolished, E-PHYS-STATELAW-001).

    Notes
    -----
    STATE LAW (E-PHYS-STATELAW-001):
      - L (dried-layer thickness from the top) is the unified state variable,
        clamped to [0, H].
      - Remaining ice mass is DERIVED, never integrated:
            ice_mass(L) = RHO_ICE * V(H - L)
        At L=0 this equals RHO_ICE*V(H); at L>=H it is ~0 (<= 1% initial).
      - Simulated endpoint = first timestep with L >= H. The integration
        TERMINATES there; later timesteps are post-primary and masked out.
      - NoRootError from solve_Tp propagates (no silent Tp := Ts - 5 pin).
    """
    n = len(t_s)
    Tp_sim = np.full(n, np.nan)
    valid_mask = np.zeros(n, dtype=bool)
    fill_height = tube.fill_height(fill_volume_m3)  # H (4.0 mm anchor in anchored mode)
    contact_area = tube.contact_area_m2(fill_height)
    L = 0.0  # UNIFIED STATE VARIABLE: dried-layer thickness from the top (m),
             # always clamped to [0, H]. Ice mass is DERIVED from L only:
             #   ice_mass = rho_ice * V(H - L)   (remaining mass, never accumulated)
    terminated = False  # True once the simulated endpoint (L >= H) is reached

    def _derived_ice_mass() -> float:
        """REMAINING ice mass (kg) consistent with the current front depth L."""
        if isinstance(tube, AnchoredTubeGeometry):
            return tube.derived_ice_mass_kg(L, RHO_ICE)
        # Legacy frustum fallback: remaining column volume below the front.
        remaining_h = max(fill_height - L, 0.0)
        r_bot = tube.bottom_radius_m
        r_top = tube.radius_at_height(remaining_h)
        v_remaining = (np.pi * remaining_h / 3.0) * (r_bot ** 2 + r_bot * r_top + r_top ** 2)
        return float(RHO_ICE * v_remaining)

    # Initialize for transient mode
    if use_transient:
        ice_mass = _derived_ice_mass()  # DERIVED from L, not an independent state
        Tp_prev = Ts_k[0] - 5.0  # Initial guess for product temp
    else:
        ice_mass = None
        Tp_prev = None

    for i in range(n):
        if terminated:
            # Post-primary timestep: the tube is empty (L >= H). The
            # primary-drying model must NEVER simulate through an empty tube,
            # so Tp stays NaN and the timestep is excluded from residuals.
            break

        Rp = Rp0 + A1 * L / (1.0 + A2 * L)
        A_front = tube.front_area_m2(fill_height, L)

        if use_transient and i > 0:
            dt = float(t_s[i] - t_s[i - 1])
            flux, Tp_k = solve_Tp(
                float(Ts_k[i]), float(Pc_pa[i]), float(Rp), float(Rs),
                float(contact_area), float(A_front), float(Kv),
                Tp_prev_k=float(Tp_prev), dt=dt, ice_mass=float(ice_mass)
            )
            Tp_prev = Tp_k
        else:
            flux, Tp_k = solve_Tp(
                float(Ts_k[i]), float(Pc_pa[i]), float(Rp), float(Rs),
                float(contact_area), float(A_front), float(Kv)
            )

        Tp_sim[i] = Tp_k
        valid_mask[i] = True

        if i < n - 1:
            dt = float(t_s[i + 1] - t_s[i])
            # The dried layer grows by the sublimed ice volume divided by the
            # LOCAL front area A(front height). With the corrected state law
            # ice_mass = rho*V(H-L) this exactly conserves mass: the volume
            # removed from the frozen column equals flux*dt/rho.
            if A_front > 0.0:
                L += max(flux * dt / RHO_ICE, 0.0) / A_front
            # CLAMP L to [0, H] (defect 2): L can no longer be integrated past
            # the fill height. The FIRST time L reaches H IS the simulated
            # endpoint; terminate the primary-drying integration there.
            L = min(max(L, 0.0), fill_height)
            if L >= fill_height:
                terminated = True
            if use_transient:
                ice_mass = _derived_ice_mass()  # re-DERIVE, never accumulate

    if return_masks:
        return Tp_sim, valid_mask, terminated
    return Tp_sim


def _first_endpoint_time_s(
    L_trajectories: list[np.ndarray],
    t_arrays: list[np.ndarray],
    H_m: float,
) -> float | None:
    """Simulated endpoint = FIRST time the dried-layer depth L reaches/exceeds
    the total ice-column height H (the measured 4.0 mm anchor in anchored mode).

    NO extrapolation fallback exists: if L < H throughout the entire simulated
    window this returns None and callers must report the endpoint as
    "not reached within the simulated window".
    """
    cumulative = 0.0
    for t_arr, L_traj in zip(t_arrays, L_trajectories):
        if len(L_traj) == 0:
            continue
        hit = np.where(np.asarray(L_traj) >= H_m)[0]
        if hit.size:
            return cumulative + float(t_arr[int(hit[0])])
        if len(t_arr) > 0:
            cumulative += float(t_arr[-1])
    return None


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
    _shelf_temp_std: float = 0.0  # Measured shelf temperature std dev (K) for reporting
    _pressure_std: float = 0.0  # Measured pressure std dev (Pa) for reporting
    _median_probe_spread_k: float | None = None  # Median (max-min of TP01..TP04) probe spread (K)


def simulate_continuous_primary_drying(
    segments: list[DryingSegment],
    Kv: float, Rp0: float, A1: float, A2: float, Rs: float = 0.0,
    use_hybrid: bool = False
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray],
           list[np.ndarray], dict]:
    """Simulate primary drying continuously across multiple segments (Steps 2-9).

    This maintains continuity of:
    - Dried layer thickness L (ice front position), CLAMPED to [0, H]
      (E-PHYS-STATELAW-001 defect 2: the primary-drying model never
      integrates past the fill height and never simulates through an
      empty tube)
    - Ice mass — DERIVED from L only: ice_mass = RHO_ICE * V(H - L)
      (remaining mass; E-PHYS-STATELAW-001 defect 1 corrected the earlier
      inverted derivation rho*(V(H)-V(H-L)) which returned SUBLIMED mass)
    - Product temperature Tp

    Across all segments, applying appropriate physics (steady-state vs transient)
    based on whether each segment is a Hold or Ramp step.

    Parameters
    ----------
    segments : list[DryingSegment]
        List of consecutive drying segments (must be in chronological order)
    Kv, Rp0, A1, A2 : float
        Fitted model parameters
    Rs : float, default 0.0
        Stopper/chamber resistance
    use_hybrid : bool, default False
        If True, use steady-state for Hold steps and transient for Ramp steps.
        If False, use transient mode for all segments.

    Returns
    -------
    Tp_simulations : list[np.ndarray]
        List of simulated product temperature arrays, one per segment.
        Post-primary timesteps (at/after the first time L >= H) and timesteps
        where the heat/mass balance had no root carry NaN — they must be
        excluded from residual computations.
    t_arrays : list[np.ndarray]
        List of time arrays (seconds from segment start), one per segment
    ice_mass_trajectories : list[np.ndarray]
        List of REMAINING ice-mass trajectories (kg), one per segment. These
        are DERIVED from the unified state L at each timestep (rho*V(H-L)) —
        never an independently accumulated quantity.
    front_depth_trajectories : list[np.ndarray]
        List of dried-layer depth L(t) (m), one per segment, always within
        [0, H]. The simulated endpoint is defined on this trajectory: first
        time L >= H.
    diagnostics : dict
        Exclusion bookkeeping (E-PHYS-STATELAW-001):
        - "n_post_primary_excluded": {label: int} — timesteps dropped because
          the tube was already empty (L >= H reached earlier).
        - "n_no_root_excluded": {label: int} — timesteps dropped because the
          heat/mass balance had NO root (NoRootError from solve_Tp; silent
          Tp := Ts - 5.0 pinning is abolished).
        - "segment_lengths": {label: int} — total timesteps per segment.
        - "hold_flags": {label: bool} — whether the segment is a Hold step.
        - "endpoint_reached": bool — True once L >= H occurred anywhere.
        - "endpoint_time_s": float | None — cumulative time of first L >= H.
    """
    # Initialize state variables at the start of primary drying
    # These will be carried forward across all segments
    L = 0.0  # UNIFIED STATE VARIABLE: dried layer thickness from top (m),
             # clamped to [0, H] after every update.
    tube = segments[0].tube
    fill_volume_m3 = segments[0].fill_volume_m3
    # H: total ice-column height at the start of primary drying. For the
    # anchored geometry this is the measured anchor height (4.0 mm); the
    # simulated endpoint is defined as the first time L >= H.
    fill_height = tube.fill_height(fill_volume_m3)
    contact_area = tube.contact_area_m2(fill_height)

    def _derived_ice_mass(L_now: float) -> float:
        """REMAINING ice mass (kg) DERIVED from the front depth L: rho*V(H-L)."""
        if isinstance(tube, AnchoredTubeGeometry):
            return tube.derived_ice_mass_kg(L_now, RHO_ICE)
        remaining_h = max(fill_height - L_now, 0.0)
        r_bot = tube.bottom_radius_m
        r_top = tube.radius_at_height(remaining_h)
        v_remaining = (np.pi * remaining_h / 3.0) * (r_bot ** 2 + r_bot * r_top + r_top ** 2)
        return float(RHO_ICE * v_remaining)

    initial_ice_mass = _derived_ice_mass(0.0)
    ice_mass = initial_ice_mass  # derived, not an independent state

    # Initial product temperature guess
    Tp_prev = segments[0].Ts_k[0] - 5.0  # Start ~5K below shelf temp

    all_Tp_sim = []
    all_t_arrays = []
    all_ice_mass_trajectories = []
    all_L_trajectories = []

    # --- exclusion bookkeeping (E-PHYS-STATELAW-001) --------------------
    def _seg_label(idx: int, seg: DryingSegment) -> str:
        return seg.label or f"segment_{idx}"

    n_post_primary_excluded: dict[str, int] = {}
    n_no_root_excluded: dict[str, int] = {}
    segment_lengths: dict[str, int] = {}
    hold_flags: dict[str, bool] = {}

    terminated = False          # once L >= H: post-primary, stop simulating
    cumulative_time_s = 0.0     # time since start of primary drying
    endpoint_time_s: float | None = None

    for seg_idx, seg in enumerate(segments):
        label = _seg_label(seg_idx, seg)
        n = len(seg.t_s)
        Tp_sim = np.full(n, np.nan)
        ice_mass_trajectory = np.full(n, np.nan)  # derived from L at each timestep
        L_trajectory = np.full(n, np.nan)
        n_post_primary_excluded[label] = 0
        n_no_root_excluded[label] = 0
        segment_lengths[label] = n
        hold_flags[label] = bool(seg.is_hold_step)

        # Debug print for state continuity verification
        print(f"[STATE-CONTINUITY] Segment {seg_idx} ({label}): L_start = {L:.6f} m, ice_mass = {ice_mass:.9f} kg")

        # Determine physics mode for this segment
        seg_use_transient = True  # Default to transient for continuity
        if use_hybrid:
            # Hold steps: steady-state (but still continue L and derived ice_mass)
            # Ramp steps: transient
            seg_use_transient = not seg.is_hold_step

        for i in range(n):
            if terminated:
                # POST-PRIMARY (defect 2): the tube is empty (L >= H was
                # reached). The primary-drying model must NEVER simulate
                # through an empty tube. Mark this timestep post-primary,
                # exclude it from residuals (NaN), and count it.
                n_post_primary_excluded[label] += 1
                continue

            # Calculate product resistance based on current dried layer thickness
            Rp = Rp0 + A1 * L / (1.0 + A2 * L)

            # Calculate front area based on current front position
            A_front = tube.front_area_m2(fill_height, L)

            # Record DERIVED ice mass at this timestep
            ice_mass = _derived_ice_mass(L)
            ice_mass_trajectory[i] = ice_mass
            L_trajectory[i] = L

            t_abs = cumulative_time_s + float(seg.t_s[i])

            try:
                if seg_use_transient and i > 0:
                    # Transient mode: include heat accumulation term
                    dt = float(seg.t_s[i] - seg.t_s[i - 1])
                    flux, Tp_k = solve_Tp(
                        float(seg.Ts_k[i]), float(seg.Pc_pa[i]), float(Rp), float(Rs),
                        float(contact_area), float(A_front), float(Kv),
                        Tp_prev_k=float(Tp_prev), dt=dt, ice_mass=float(ice_mass)
                    )
                    Tp_prev = Tp_k
                else:
                    # Steady-state mode (for Hold steps in hybrid mode)
                    # Still uses current L (and the ice mass derived from it)
                    flux, Tp_k = solve_Tp(
                        float(seg.Ts_k[i]), float(seg.Pc_pa[i]), float(Rp), float(Rs),
                        float(contact_area), float(A_front), float(Kv)
                    )
                    Tp_prev = Tp_k  # Still update for continuity
            except NoRootError:
                # DEFECT 3: the heat/mass balance has NO root for Tp. Do NOT
                # silently pin Tp := Ts - 5.0 (that manufactured the constant
                # Hold-step bias). Exclude this timestep from residuals and
                # count it. State L does not advance (flux unknown).
                Tp_sim[i] = np.nan
                n_no_root_excluded[label] += 1
                continue

            Tp_sim[i] = Tp_k

            # Simulated endpoint check AT THIS TIMESTEP: if the front has
            # arrived at the tube bottom (L >= H), this is the endpoint.
            if L >= fill_height:
                terminated = True
                if endpoint_time_s is None:
                    endpoint_time_s = t_abs

            # Update the unified state variable L for the next timestep.
            # Volume sublimed over dt divided by the LOCAL front area keeps
            # ice_mass == rho*V(H-L) exactly consistent (no separate
            # ice-mass accumulation exists anywhere).
            if i < n - 1 and not terminated:
                dt = float(seg.t_s[i + 1] - seg.t_s[i])
                if A_front > 0.0:
                    L += max(flux * dt / RHO_ICE, 0.0) / A_front
                # CLAMP L to [0, H] (defect 2): L can never exceed the fill
                # height. The first time L reaches H IS the simulated
                # endpoint; everything after is post-primary.
                L = min(max(L, 0.0), fill_height)
                if L >= fill_height:
                    terminated = True
                    if endpoint_time_s is None:
                        # Endpoint occurs at the NEXT timestep's absolute time.
                        endpoint_time_s = t_abs + dt

        cumulative_time_s += float(seg.t_s[-1]) if n > 0 else 0.0

        all_Tp_sim.append(Tp_sim)
        all_t_arrays.append(seg.t_s.copy())
        all_ice_mass_trajectories.append(ice_mass_trajectory)
        all_L_trajectories.append(L_trajectory)

    diagnostics = {
        "n_post_primary_excluded": n_post_primary_excluded,
        "n_no_root_excluded": n_no_root_excluded,
        "segment_lengths": segment_lengths,
        "hold_flags": hold_flags,
        "endpoint_reached": bool(terminated),
        "endpoint_time_s": endpoint_time_s,
    }

    return all_Tp_sim, all_t_arrays, all_ice_mass_trajectories, all_L_trajectories, diagnostics


def fit_parameters_joint(
    segments: list[DryingSegment], 
    initial_guess: dict | None = None,
    use_transient: bool = False,
    use_hybrid: bool = False,
    endpoint_time_s: float | None = None,
    mass_balance_weight: float | None = None
) -> dict:
    """Fit ONE shared {Kv, Rp0, A1, A2} across multiple segments.
    
    When use_hybrid=True or use_transient=True, the simulation maintains continuity
    of dried layer thickness L, ice mass, and product temperature across all segments
    (Steps 2-9 within Phase 6), applying appropriate physics (steady-state vs transient)
    based on Hold/Ramp detection.
    
    Parameters
    ----------
    segments : list[DryingSegment]
        List of drying segments to fit jointly (should be in chronological order)
    initial_guess : dict, optional
        Initial guess for parameters
    use_transient : bool, default False
        If True, maintain continuity of state variables across segments using
        transient heat accumulation throughout
    use_hybrid : bool, default False
        If True, apply quasi-steady state for Hold steps (detected by shelf_setpt stability)
        and transient mode for Ramp steps, while maintaining state continuity across all steps.
        This overrides use_transient when enabled.
    endpoint_time_s : float, optional
        Detected primary-drying endpoint time in seconds from start of primary drying.
        If provided, a mass-balance constraint is added to enforce that simulated ice
        depletes at this detected endpoint.
    mass_balance_weight : float, optional
        Weight for the mass-balance residual term in the joint fit.
        If None (default), the weight is AUTO-scaled inside resid() to
        sqrt(N_temperature_residuals), so that a 100% endpoint error costs
        roughly the same as the full temperature misfit. A single scalar
        residual with weight 1.0 is negligible against ~1000 temperature
        residuals and cannot move the optimizer.

    Returns
    -------
    fitted : dict
        Dictionary containing fitted parameters and diagnostics:
        - Kv, Rp0, A1, A2: Fitted model parameters
        - rms_error_k: Overall RMS error in K
        - converged: Boolean indicating convergence
        - per_segment_rms_k: Per-segment RMS errors
        - simulated_endpoint_time_s: Simulated endpoint time (if endpoint_time_s provided)
        - mass_balance_residual: Mass-balance residual value (if endpoint_time_s provided)
        - mass_balance_penalty_share: Fraction of the final total sum of squares
          contributed by the squared mass-balance residual (None if inactive)
    """
    guess = initial_guess or dict(Kv=15.0, Rp0=2e4, A1=1e6, A2=100.0)
    x0 = np.array([guess["Kv"], guess["Rp0"], guess["A1"], guess["A2"]], dtype=float)
    
    # Store endpoint_time_s in outer scope for use in resid function
    _endpoint_time_s = endpoint_time_s
    _mass_balance_weight = mass_balance_weight  # None => AUTO scaling (see resid)

    # Track whether the mass-balance penalty branch actually executed during the fit
    _mass_balance_state = {"active": False, "weight_used": None}

    # H: total ice-column height at start of primary drying. In anchored mode
    # this is the measured 4.0 mm anchor (fill_height() is pinned to it).
    _H_m = segments[0].tube.fill_height(segments[0].fill_volume_m3)

    def compute_simulated_endpoint_time(front_depth_trajectories: list[np.ndarray],
                                        t_arrays: list[np.ndarray]) -> float | None:
        """Compute simulated endpoint time from the unified front-depth state L.

        Delegates to _first_endpoint_time_s: SIMULATED ENDPOINT = first time
        L >= H (the measured 4.0 mm fill height in anchored mode). There is
        NO extrapolation fallback: if L never reaches H within the simulated
        window this returns None and callers must report "not reached".
        """
        return _first_endpoint_time_s(front_depth_trajectories, t_arrays, _H_m)

    def resid(x: np.ndarray) -> np.ndarray:
        Kv, Rp0, A1, A2 = x
        # Use continuous simulation across all segments
        if use_transient or use_hybrid:
            all_Tp_sim, all_t_arrays, all_ice_mass_trajectories, all_L_trajectories = \
                simulate_continuous_primary_drying(
                    segments, Kv, Rp0, A1, A2, use_hybrid=use_hybrid
                )
            out = [all_Tp_sim[i] - segments[i].Tp_measured_k for i in range(len(segments))]

            # Number of temperature residuals across all concatenated segments.
            n_temperature_residuals = sum(len(r) for r in out)

            # Add mass-balance residual if endpoint_time_s is provided
            if _endpoint_time_s is not None and _endpoint_time_s > 0:
                _mass_balance_state["active"] = True

                # AUTO weight scaling: when mass_balance_weight is None, scale the
                # single scalar mass residual by sqrt(N_temperature_residuals) so a
                # 100% endpoint error costs roughly the same as the full temperature
                # misfit (a weight of 1.0 is negligible against ~1000 temperature
                # residuals and cannot move the optimizer).
                effective_weight = _mass_balance_weight
                if effective_weight is None:
                    effective_weight = float(np.sqrt(max(n_temperature_residuals, 1)))
                _mass_balance_state["weight_used"] = effective_weight

                # Simulated endpoint = first time L >= H. If the front never
                # fully recedes within the simulated window there is NO
                # extrapolation fallback: the constraint penalizes the full
                # remaining simulated window instead (endpoint treated as
                # occurring no earlier than the end of the window).
                simulated_endpoint = compute_simulated_endpoint_time(
                    all_L_trajectories, all_t_arrays
                )
                if simulated_endpoint is None:
                    total_window_s = sum(
                        float(t_arr[-1]) for t_arr in all_t_arrays if len(t_arr) > 0
                    )
                    simulated_endpoint = total_window_s

                # Normalized mass-balance residual
                mass_residual = effective_weight * (simulated_endpoint - _endpoint_time_s) / _endpoint_time_s
                out.append(np.array([mass_residual]))
        else:
            # Legacy mode: independent segments (no state continuity)
            out = []
            for seg in segments:
                Tp_sim = simulate_cycle(
                    seg.t_s, seg.Ts_k, seg.Pc_pa, seg.tube,
                    seg.fill_volume_m3, Kv, Rp0, A1, A2, use_transient=False
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

    # Mass-balance closure diagnostics (always present in the result dict).
    # mass_balance_active is True only when endpoint_time_s was provided AND the
    # penalty branch inside resid() actually executed during the fit.
    fitted["detected_endpoint_time_s"] = endpoint_time_s if (endpoint_time_s is not None and endpoint_time_s > 0) else None
    fitted["simulated_endpoint_time_s"] = None
    fitted["mass_balance_residual"] = None
    fitted["mass_balance_active"] = bool(_mass_balance_state["active"])
    fitted["mass_balance_penalty_share"] = None

    # Calculate per-segment RMS errors, residual-sign / probe-spread
    # diagnostics, and mass-balance diagnostics using continuous simulation.
    per_segment = {}
    per_segment_diagnostics: dict[str, dict] = {}
    Kv, Rp0, A1, A2 = x_vals

    def _fill_residual_diags(label: str, resid_arr: np.ndarray, seg) -> None:
        """Record mean/median residual sign (simulated minus measured) and the
        median probe spread (max-min of TP01..TP04) for one segment."""
        mean_r = float(np.mean(resid_arr))
        median_r = float(np.median(resid_arr))
        per_segment_diagnostics[label] = {
            "mean_residual_k": mean_r,
            "median_residual_k": median_r,
            "residual_sign": (
                "positive" if median_r > 0 else
                "negative" if median_r < 0 else "zero"
            ),
            "median_probe_spread_k": getattr(seg, "_median_probe_spread_k", None),
        }

    if use_transient or use_hybrid:
        all_Tp_sim, all_t_arrays, all_ice_mass_trajectories, all_L_trajectories = \
            simulate_continuous_primary_drying(
                segments, Kv, Rp0, A1, A2, use_hybrid=use_hybrid
            )
        for i, seg in enumerate(segments):
            label = seg.label or f"segment_{id(seg)}"
            resid_i = all_Tp_sim[i] - seg.Tp_measured_k
            per_segment[label] = float(np.sqrt(np.mean(resid_i ** 2)))
            _fill_residual_diags(label, resid_i, seg)

        # Compute mass-balance diagnostics if endpoint was provided
        if endpoint_time_s is not None and endpoint_time_s > 0:
            tube = segments[0].tube
            fill_volume_m3 = segments[0].fill_volume_m3
            # Initial ice mass DERIVED from the anchored geometry when
            # available: rho * V(H); legacy fallback keeps rho * fill volume.
            if isinstance(tube, AnchoredTubeGeometry):
                initial_ice_mass = tube.derived_ice_mass_kg(0.0, RHO_ICE)
            else:
                initial_ice_mass = RHO_ICE * fill_volume_m3

            # Simulated endpoint = first time L >= H. NO extrapolation
            # fallback: if never reached within the window, report None.
            simulated_endpoint = compute_simulated_endpoint_time(
                all_L_trajectories, all_t_arrays
            )
            fitted["endpoint_reached_in_window"] = simulated_endpoint is not None

            # Resolve the effective penalty weight at the final parameters.
            # mass_balance_weight=None means AUTO: sqrt(N_temperature_residuals),
            # computed from the concatenated segment lengths (same rule as resid()).
            n_temperature_residuals = sum(len(seg.t_s) for seg in segments)
            effective_weight = mass_balance_weight
            if effective_weight is None:
                effective_weight = float(np.sqrt(max(n_temperature_residuals, 1)))
            if _mass_balance_state.get("weight_used") is not None:
                effective_weight = float(_mass_balance_state["weight_used"])

            endpoint_for_residual = simulated_endpoint
            if endpoint_for_residual is None:
                # Endpoint NOT reached: no extrapolation. Penalize against the
                # end of the simulated window (a strict lower bound).
                endpoint_for_residual = sum(
                    float(t_arr[-1]) for t_arr in all_t_arrays if len(t_arr) > 0
                )
            mass_residual_final = effective_weight * (endpoint_for_residual - endpoint_time_s) / endpoint_time_s

            fitted["simulated_endpoint_time_s"] = simulated_endpoint
            fitted["detected_endpoint_time_s"] = endpoint_time_s
            fitted["mass_balance_residual"] = mass_residual_final

            # Penalty share of cost: (mass_residual**2) / (total sum of squares),
            # evaluated at the FINAL fitted parameters. Uses the optimizer's final
            # residual vector when available (temperature SS + squared mass term).
            temp_ss_final = float(
                sum(np.sum((all_Tp_sim[i] - segments[i].Tp_measured_k) ** 2)
                    for i in range(len(segments)))
            )
            mass_term_for_ss = mass_residual_final
            try:
                final_fun = np.asarray(getattr(result, "fun", None), dtype=float)
                if final_fun is not None and len(final_fun) == n_temperature_residuals + 1:
                    mass_term_for_ss = float(final_fun[-1])
                    temp_ss_final = float(np.sum(final_fun[:-1] ** 2))
            except Exception:
                pass
            total_ss = temp_ss_final + mass_term_for_ss ** 2
            fitted["mass_balance_penalty_share"] = (
                float(mass_term_for_ss ** 2 / total_ss) if total_ss > 0 else 0.0
            )

            # Store detailed mass-balance physics for reporting
            fitted["initial_ice_mass_kg"] = initial_ice_mass

            # Get final ice mass from end of last segment (DERIVED trajectory)
            if all_ice_mass_trajectories and len(all_ice_mass_trajectories[-1]) > 0:
                fitted["final_ice_mass_kg"] = float(all_ice_mass_trajectories[-1][-1])

            # Compute mean ice depletion rate from final segment
            if len(all_ice_mass_trajectories) > 0 and len(all_ice_mass_trajectories[-1]) >= 2:
                final_seg_ice = all_ice_mass_trajectories[-1]
                final_seg_t = all_t_arrays[-1]
                dt = final_seg_t[-1] - final_seg_t[0]
                if dt > 0:
                    depletion_rate = (final_seg_ice[0] - final_seg_ice[-1]) / dt
                    fitted["ice_depletion_rate_kg_s"] = float(depletion_rate)
    else:
        # Legacy mode: independent segments
        for seg in segments:
            Tp_sim = simulate_cycle(
                seg.t_s, seg.Ts_k, seg.Pc_pa, seg.tube,
                seg.fill_volume_m3, Kv, Rp0, A1, A2, use_transient=False
            )
            label = seg.label or f"segment_{id(seg)}"
            resid_i = Tp_sim - seg.Tp_measured_k
            per_segment[label] = float(np.sqrt(np.mean(resid_i ** 2)))
            _fill_residual_diags(label, resid_i, seg)

    fitted["per_segment_rms_k"] = per_segment
    fitted["per_segment_diagnostics"] = per_segment_diagnostics
    return fitted

if __name__ == "__main__":
    print(__doc__)