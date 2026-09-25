"""
Core pipeline physics module - single source of truth for lyophilization digital twin.

This module contains the canonical, validated implementations of all pipeline
physics functions. GUI applications and other drivers MUST import from this module
to ensure consistent behavior across all interfaces.

Event: E-ARCH-CORE-001
Phase: Phase 0
"""

import sys
from typing import Optional

import numpy as np
import pandas as pd

from geometry import PCRTubeGeometry
from pikal_model import DryingSegment


REQUIRED_MODEL_COLUMNS = [
    "Phase",
    "Cycle",
    "Step",
    "timestamp",
    "shelf_temp_k",
    "pressure_pa",
    "product_temp_k",
]


class CalibratedPCRTubeGeometry(PCRTubeGeometry):
    """
    PCR tube geometry that uses empirical fill-height calibration if provided.

    pikal_model.simulate_cycle() internally calls:

        tube.fill_height(fill_volume_m3)

    By overriding fill_height() here, the calibration points supplied through
    --calib are actually used in the model.
    """

    def __init__(self, calibration_points=None, contact_efficiency=0.22, **kwargs):
        super().__init__(contact_efficiency=contact_efficiency, **kwargs)
        self.calibration_points = calibration_points or []

    def fill_height(self, fill_volume_m3):
        if not self.calibration_points:
            return super().fill_height(fill_volume_m3)

        vols = np.array([vol for vol, _ in self.calibration_points], dtype=float)

        if vols.size:
            fill_vol_ul = fill_volume_m3 * 1e9
            min_cal_ul = vols.min() * 1e9
            max_cal_ul = vols.max() * 1e9

            if fill_vol_ul < min_cal_ul or fill_vol_ul > max_cal_ul:
                print(
                    f"[warn] Fill volume {fill_vol_ul:.3f} uL is outside calibration range "
                    f"{min_cal_ul:.3f}-{max_cal_ul:.3f} uL. "
                    "Interpolation will clamp to the nearest calibration endpoint.",
                    file=sys.stderr,
                )

        return self.fill_height_from_calibration(
            fill_volume_m3,
            self.calibration_points,
        )


def parse_calib(calib_input):
    """
    Parse calibration input. Accepts either a list of strings (from CLI)
    or a single string (from GUI).
    
    Format: "vol_ul:height_mm" (e.g., "6:1.8 12:2.9 20:4.1")
    """
    if not calib_input:
        return []
    
    # Handle both string input (GUI) and list input (CLI)
    if isinstance(calib_input, str):
        calib_args = calib_input.strip().split()
    elif isinstance(calib_input, list):
        calib_args = calib_input
    else:
        raise ValueError(f"Invalid calibration input type: {type(calib_input)}")
    
    if not calib_args:
        return []

    points = []

    for arg in calib_args:
        try:
            vol_ul_str, height_mm_str = arg.split(":", maxsplit=1)
            vol_ul = float(vol_ul_str)
            height_mm = float(height_mm_str)
        except Exception as exc:
            raise ValueError(
                f"Invalid calibration value {arg!r}. "
                "Expected format vol_ul:height_mm, e.g. 6:1.8"
            ) from exc

        if vol_ul <= 0:
            raise ValueError(
                f"Invalid calibration volume in {arg!r}. Volume must be positive."
            )

        if height_mm <= 0:
            raise ValueError(
                f"Invalid calibration height in {arg!r}. Height must be positive."
            )

        vol_m3 = vol_ul * 1e-9
        height_m = height_mm * 1e-3
        points.append((vol_m3, height_m))

    if not points:
        return []

    # Sort by volume and average duplicate volume entries.
    by_volume = {}

    for vol_m3, height_m in points:
        by_volume.setdefault(vol_m3, []).append(height_m)

    cleaned = [
        (vol_m3, float(np.mean(heights)))
        for vol_m3, heights in sorted(by_volume.items())
    ]

    return cleaned

def coerce_phase_code(df, raw_code):
    """
    Convert the user-supplied phase code into something comparable to df['Phase'].
    """
    raw = str(raw_code).strip()

    if "Phase" not in df.columns:
        raise ValueError("Missing required column: Phase")

    phase_values = df["Phase"].dropna()

    # Numeric Phase column.
    if phase_values.dtype.kind in "if":
        try:
            numeric_raw = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"The Phase column is numeric, but --primary-phase-code "
                f"{raw_code!r} is not numeric."
            ) from exc

        if numeric_raw.is_integer():
            return int(numeric_raw)

        return numeric_raw

    # Object/string/category Phase column.
    unique_phases = phase_values.unique()

    # Exact string match.
    if raw in unique_phases:
        return raw

    # Try integer conversion.
    try:
        as_int = int(raw)
        if as_int in unique_phases:
            return as_int
    except ValueError:
        pass

    # Try float conversion.
    try:
        as_float = float(raw)
        if as_float in unique_phases:
            return as_float
    except ValueError:
        pass

    # Case-insensitive string match.
    raw_lower = raw.lower()

    for value in unique_phases:
        if isinstance(value, str) and value.strip().lower() == raw_lower:
            return value

    # Let the later filter fail with a helpful message if nothing matches.
    return raw


def choose_pressure_column(df):
    """
    Choose chamber pressure column.

    Preferred order:
      1. capman_pa
      2. pirani_pa
      3. vacuum_pa
      4. vac_setpt_pa

    Writes result to df['pressure_pa'].
    """
    candidates = [
        "capman_pa",
        "pirani_pa",
        "vacuum_pa",
        "vac_setpt_pa",
    ]

    for col in candidates:
        if col in df.columns and df[col].notna().any():
            df["pressure_pa"] = df[col]

            if col != "capman_pa":
                print(
                    f"[warn] capman_pa not usable; using {col} for chamber pressure. "
                    "Pirani readings are water-vapor dependent and are less ideal "
                    "for chamber pressure during primary drying.",
                    file=sys.stderr,
                )

            return col

    raise ValueError(
        "No usable pressure column found. Expected at least one of: "
        + ", ".join(candidates)
    )


def choose_product_temperature_column(df, use_min=False):
    """
    Choose product temperature column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data frame with temperature columns.
    use_min : bool, optional
        If True, use the minimum probe reading (prod_min_k) to approximate
        the ice-front temperature when probes are fully immersed.
        If False (default), use the average (prod_avg_k).

    Preferred:
      - If use_min=True: prod_min_k
      - If use_min=False: prod_avg_k

    Fallback:
      mean of TPxx_k columns (or min if use_min=True)

    Writes result to df['product_temp_k'].
    """
    if use_min:
        if "prod_min_k" in df.columns and df["prod_min_k"].notna().any():
            df["product_temp_k"] = df["prod_min_k"]
            return "prod_min_k"

        tp_cols = [
            col
            for col in df.columns
            if col.lower().startswith("tp") and col.lower().endswith("_k")
        ]

        if tp_cols:
            df["product_temp_k"] = df[tp_cols].min(axis=1)

            print(
                f"[warn] prod_min_k not usable; using min of {tp_cols} "
                "as product temperature (ice-front approximation).",
                file=sys.stderr,
            )

            return f"min({', '.join(tp_cols)})"
    else:
        if "prod_avg_k" in df.columns and df["prod_avg_k"].notna().any():
            df["product_temp_k"] = df["prod_avg_k"]
            return "prod_avg_k"

        tp_cols = [
            col
            for col in df.columns
            if col.lower().startswith("tp") and col.lower().endswith("_k")
        ]

        if tp_cols:
            df["product_temp_k"] = df[tp_cols].mean(axis=1)

            print(
                f"[warn] prod_avg_k not usable; using mean of {tp_cols} "
                "as product temperature.",
                file=sys.stderr,
            )

            return f"mean({', '.join(tp_cols)})"

    raise ValueError(
        "No usable product temperature column found. "
        "Expected prod_avg_k/prod_min_k or TPxx_k columns."
    )


def find_endpoint(
    df,
    rolling_window_minutes: float = 30.0,
    sustain_minutes: float = 30.0,
    min_baseline_fraction: float = 0.25,
):
    """
    Detect the primary drying endpoint using Pirani decline transition.
    
    CRITICAL PHYSICS: In this system, Pirani and CM never fully converge.
    During primary drying, Pirani reads ~5.3 Pa (40 mTorr) higher than CM due to water vapor.
    After primary drying, Pirani drops and stabilizes at ~1.3 Pa (10 mTorr) above CM.
    The correct endpoint is when Pirani drops to this new baseline and stabilizes.
    
    Algorithm:
    1. Calculate Pirani-CM difference: diff = pirani_pa - capman_pa
    2. Compute rolling mean of difference over configurable window to smooth noise
    3. Determine "primary drying baseline" as median diff during first 25% of data
    4. Determine "post-drying baseline" as median diff during last 25% of data
    5. Set transition threshold as midpoint between baselines
    6. Endpoint is FIRST timestamp where rolling mean drops below threshold AND
       remains below for sustained duration
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp', 'pirani_pa', and 'capman_pa' columns.
    rolling_window_minutes : float, default 30.0
        Window size for rolling mean smoothing in minutes.
    sustain_minutes : float, default 30.0
        Minimum duration (in minutes) the condition must be met continuously.
    min_baseline_fraction : float, default 0.25
        Fraction of data used to compute primary and post-drying baselines.
    
    Returns
    -------
    timestamp : pd.Timestamp or None
        The exact timestamp when the endpoint was detected (start of sustained transition).
        If no sustained transition is found, returns the last timestamp of the primary phase
        and logs a warning.
    diagnostics : dict
        Dictionary containing:
        - 'primary_baseline': Median Pirani-CM diff during early phase (Pa)
        - 'post_baseline': Median Pirani-CM diff during late phase (Pa)
        - 'threshold': Transition threshold used (Pa)
        - 'detected': Boolean indicating if transition was detected
    """
    # Initialize diagnostics
    diagnostics = {
        'primary_baseline': None,
        'post_baseline': None,
        'threshold': None,
        'detected': False,
    }
    
    if "pirani_pa" not in df.columns or "capman_pa" not in df.columns:
        print(
            "[warn] Cannot detect endpoint: missing pirani_pa or capman_pa columns. "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        if "timestamp" in df.columns and len(df) > 0:
            return df["timestamp"].iloc[-1], diagnostics
        return None, diagnostics
    
    # Sort by timestamp and reset index
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    
    # Ensure we have valid data
    valid_mask = (
        df_sorted["pirani_pa"].notna() 
        & df_sorted["capman_pa"].notna() 
        & (df_sorted["capman_pa"] > 0)
    )
    
    if not valid_mask.any():
        print(
            "[warn] No valid Pirani/Capman pressure readings for endpoint detection. "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        if "timestamp" in df_sorted.columns and len(df_sorted) > 0:
            return df_sorted["timestamp"].iloc[-1], diagnostics
        return None, diagnostics
    
    # Filter to valid rows only for calculation
    df_valid = df_sorted[valid_mask].copy().reset_index(drop=True)
    
    if len(df_valid) < 10:
        print(
            "[warn] Insufficient valid data points for endpoint detection. "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        if "timestamp" in df_sorted.columns and len(df_sorted) > 0:
            return df_sorted["timestamp"].iloc[-1], diagnostics
        return None, diagnostics
    
    # Step 1: Calculate Pirani-CM difference
    diff = df_valid["pirani_pa"] - df_valid["capman_pa"]
    
    # Step 2: Compute rolling mean of difference
    timestamps = df_valid["timestamp"]
    time_diffs = timestamps.diff().dt.total_seconds().dropna()
    median_interval_sec = time_diffs.median() if len(time_diffs) > 0 else 60.0
    
    if median_interval_sec <= 0:
        median_interval_sec = 60.0
    
    # Convert rolling window to number of samples
    rolling_window_sec = rolling_window_minutes * 60.0
    rolling_window_samples = max(int(np.ceil(rolling_window_sec / median_interval_sec)), 1)
    
    # Apply rolling mean to smooth noise
    diff_rolling = diff.rolling(window=rolling_window_samples, min_periods=1).mean()
    
    # Step 3: Determine primary drying baseline (first 25% of data)
    n_points = len(diff_rolling)
    baseline_n = max(int(np.ceil(n_points * min_baseline_fraction)), 1)
    
    primary_baseline = float(diff_rolling.iloc[:baseline_n].median())
    diagnostics['primary_baseline'] = primary_baseline
    
    # Step 4: Determine post-drying baseline (last 25% of data)
    post_baseline = float(diff_rolling.iloc[-baseline_n:].median())
    diagnostics['post_baseline'] = post_baseline
    
    # Step 5: Set transition threshold as midpoint between baselines
    threshold = (primary_baseline + post_baseline) / 2.0
    diagnostics['threshold'] = threshold
    
    print(
        f"[info] Endpoint detection: primary_baseline={primary_baseline:.3f} Pa, "
        f"post_baseline={post_baseline:.3f} Pa, threshold={threshold:.3f} Pa",
        file=sys.stderr,
    )
    
    # Sanity check: if baselines are very close or inverted, use fallback
    if primary_baseline <= post_baseline:
        print(
            "[warn] Primary baseline <= post-baseline (unexpected). "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        return df_sorted["timestamp"].iloc[-1], diagnostics
    
    # Step 6: Find first point where rolling mean drops below threshold
    below_threshold = diff_rolling < threshold
    
    if not below_threshold.any():
        print(
            "[warn] Rolling mean never drops below threshold. "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        return df_sorted["timestamp"].iloc[-1], diagnostics
    
    # Convert sustain_minutes to number of samples
    sustain_sec = sustain_minutes * 60.0
    min_sustain_points = max(int(np.ceil(sustain_sec / median_interval_sec)), 1)
    
    # Find runs of consecutive True values where diff_rolling stays below threshold
    below_values = below_threshold.values
    
    # Find all indices where condition is True
    true_indices = np.where(below_values)[0]
    
    if len(true_indices) == 0:
        print(
            "[warn] No points below threshold found. "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        return df_sorted["timestamp"].iloc[-1], diagnostics
    
    # Group consecutive indices into runs
    runs = []
    current_run = [true_indices[0]]
    
    for i in range(1, len(true_indices)):
        if true_indices[i] == true_indices[i-1] + 1:
            current_run.append(true_indices[i])
        else:
            if len(current_run) >= min_sustain_points:
                runs.append(current_run)
            current_run = [true_indices[i]]
    
    # Don't forget the last run
    if len(current_run) >= min_sustain_points:
        runs.append(current_run)
    
    if not runs:
        print(
            f"[warn] No sustained transition found (need {min_sustain_points} consecutive points). "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        return df_sorted["timestamp"].iloc[-1], diagnostics
    
    # Return the timestamp at the START of the first sustained transition period
    first_run_start_idx_in_valid = runs[0][0]
    
    # Map back to original sorted dataframe index
    # We need to find the corresponding index in df_sorted
    valid_indices = df_valid.index.tolist()
    first_run_start_idx_in_sorted = valid_indices[first_run_start_idx_in_valid]
    endpoint_timestamp = df_sorted.iloc[first_run_start_idx_in_sorted]["timestamp"]
    
    diagnostics['detected'] = True
    
    print(
        f"[info] Primary drying endpoint detected at {endpoint_timestamp}. "
        f"All data after this timestamp will be excluded from the fit.",
        file=sys.stderr,
    )
    
    return endpoint_timestamp, diagnostics


def build_segments(
    df,
    primary_phase_code,
    tube,
    fill_volume_m3,
    endpoint_timestamp,
    min_segment_points=5,
    shelf_temp_std_threshold=0.1,
    pressure_std_threshold=5.0,
):
    """
    Build one DryingSegment per Cycle/Step combination inside Primary Drying.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data frame
    primary_phase_code : any
        Phase code for primary drying
    tube : PCRTubeGeometry
        Geometry object
    fill_volume_m3 : float
        Fill volume in cubic meters
    endpoint_timestamp : pd.Timestamp
        The primary drying endpoint timestamp. All data after this is excluded.
    min_segment_points : int, default 5
        Minimum number of valid rows per segment
    shelf_temp_std_threshold : float, default 0.1 K
        Maximum standard deviation of measured shelf_temp_k to classify as Hold step.
    pressure_std_threshold : float, default 5.0 Pa
        Maximum standard deviation of measured capman_pa to classify as Hold step.
        A segment is classified as "Hold" (steady-state) ONLY if BOTH thresholds are met.
    """
    missing = [col for col in REQUIRED_MODEL_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required column(s) before segmentation: {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )

    primary_df = df[df["Phase"] == primary_phase_code].copy()

    if primary_df.empty:
        available_phases = sorted(
            str(x) for x in df["Phase"].dropna().unique()
        )

        raise ValueError(
            f"No rows found with Phase == {primary_phase_code!r}. "
            f"Available Phase values: {available_phases}"
        )

    # Note: We do NOT filter out post-endpoint data here anymore.
    # Instead, endpoint truncation is handled per-segment below to enable diagnostics.
    # The endpoint_timestamp is used to slice each segment individually.

    model_columns = [
        "timestamp",
        "shelf_temp_k",
        "pressure_pa",
        "product_temp_k",
    ]

    segments = []
    skipped = []

    for (cycle_id, step_id), group in primary_df.groupby(
        ["Cycle", "Step"],
        sort=True,
    ):
        group = group.sort_values("timestamp")
        raw_row_count = len(group)

        group = group.dropna(subset=model_columns)
        rows_after_dropna = len(group)
        
        if len(group) < min_segment_points:
            skipped.append(
                f"cycle={cycle_id}_step={step_id}: kept "
                f"{len(group)}/{raw_row_count} rows "
                f"(<{min_segment_points} valid rows)"
            )
            continue

        # Check if entire segment is post-endpoint
        if endpoint_timestamp is not None and group["timestamp"].iloc[0] > endpoint_timestamp:
            skipped.append(
                f"cycle={cycle_id}_step={step_id}: Post-endpoint (starts at {group['timestamp'].iloc[0]})"
            )
            continue

        # Detect if this is a hold step or ramp step based on MEASURED signal variance,
        # NOT setpoint stability. Recipe setpoints do not prove physical steady-state.
        # A segment is classified as "Hold" ONLY if:
        #   - std(shelf_temp_k) < shelf_temp_std_threshold (e.g., 0.1 K)
        #   - std(capman_pa) < pressure_std_threshold (e.g., 5.0 Pa)
        # Otherwise, it is classified as "Ramp" (transient).
        
        shelf_temp_std = group["shelf_temp_k"].std()
        
        # Use capman_pa for pressure variance if available, otherwise fall back to pressure_pa
        if "capman_pa" in group.columns and group["capman_pa"].notna().any():
            pressure_std = group["capman_pa"].std()
        else:
            pressure_std = group["pressure_pa"].std()
        
        # Classify as Hold step only if BOTH signals are stable
        is_hold_step = (
            shelf_temp_std < shelf_temp_std_threshold and
            pressure_std < pressure_std_threshold
        )

        t0 = group["timestamp"].iloc[0]

        t_s = (
            group["timestamp"] - t0
        ).dt.total_seconds().to_numpy(dtype=float)

        # Slice arrays to terminate exactly at endpoint if this segment contains it
        rows_dropped_endpoint = 0
        if endpoint_timestamp is not None:
            end_mask = group["timestamp"] <= endpoint_timestamp
            rows_kept_before_truncation = end_mask.sum()
            rows_dropped_endpoint = (~end_mask).sum()
            group = group[end_mask].copy()
            
            # Recalculate t_s and arrays after slicing
            if len(group) < min_segment_points:
                skipped.append(
                    f"cycle={cycle_id}_step={step_id}: kept "
                    f"{len(group)}/{raw_row_count} rows after endpoint truncation "
                    f"(<{min_segment_points} valid rows)"
                )
                continue
            
            t_s = (
                group["timestamp"] - t0
            ).dt.total_seconds().to_numpy(dtype=float)

        # Diagnostic print for endpoint truncation
        if rows_dropped_endpoint > 0:
            rows_kept = len(group)
            print(f"[DIAG] cycle={cycle_id}_step={step_id}: kept {rows_kept}/{rows_kept_before_truncation} rows (dropped {rows_dropped_endpoint} rows due to endpoint truncation)")

        # Probe spread diagnostics: max minus min across TP01..TP04 at each
        # timestep; the segment reports its MEDIAN spread (E-PHYS-GEOM-ANCHOR-001).
        probe_k_cols = [c for c in ("tp01_k", "tp02_k", "tp03_k", "tp04_k")
                        if c in group.columns]
        if not probe_k_cols:
            # tolerate alternate historian naming (TP1..TP4 -> tp1_k..tp4_k)
            probe_k_cols = [c for c in ("tp1_k", "tp2_k", "tp3_k", "tp4_k")
                            if c in group.columns]
        median_probe_spread_k = None
        if len(probe_k_cols) >= 2:
            spread = group[probe_k_cols].max(axis=1) - group[probe_k_cols].min(axis=1)
            spread = spread.dropna()
            if len(spread) > 0:
                median_probe_spread_k = float(spread.median())

        segments.append(
            DryingSegment(
                t_s=t_s,
                Ts_k=group["shelf_temp_k"].to_numpy(dtype=float),
                Pc_pa=group["pressure_pa"].to_numpy(dtype=float),
                Tp_measured_k=group["product_temp_k"].to_numpy(dtype=float),
                tube=tube,
                fill_volume_m3=fill_volume_m3,
                label=f"cycle={cycle_id}_step={step_id}",
                is_hold_step=is_hold_step,
                _shelf_temp_std=shelf_temp_std,
                _pressure_std=pressure_std,
                _median_probe_spread_k=median_probe_spread_k,
            )
        )

    if skipped:
        print("[warn] Skipped short/invalid segments:", file=sys.stderr)
        for item in skipped:
            print(f"  {item}", file=sys.stderr)

    if not segments:
        raise ValueError(
            "No usable primary-drying segments were found after filtering. "
            "Check the Phase code, Cycle/Step columns, timestamps, pressure "
            "columns, and product-temperature columns."
        )

    return segments
