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

    def __init__(self, calibration_points=None, **kwargs):
        super().__init__(**kwargs)
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


def find_endpoint(df, abs_tol_pa=1.0, rel_tol=0.15, sustain_minutes=20.0):
    """
    Detect the primary drying endpoint using Pirani vs Capacitance Manometer convergence.
    
    The endpoint is triggered when the pressure difference between Pirani and Capacitance
    Manometer readings falls below a threshold and remains there for a sustained period.
    This indicates that water vapor sublimation has ceased (ice front disappeared).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp', 'pirani_pa', and 'capman_pa' columns.
    abs_tol_pa : float, default 1.0 Pa
        Absolute tolerance for pressure difference.
    rel_tol : float, default 0.15
        Relative tolerance (fraction of capman_pa).
    sustain_minutes : float, default 20.0
        Minimum duration (in minutes) the convergence condition must be met continuously.
    
    Returns
    -------
    timestamp : pd.Timestamp or None
        The exact timestamp when the endpoint was detected (start of sustained convergence).
        If no sustained convergence is found, returns the last timestamp of the primary phase
        and logs a warning.
    """
    if "pirani_pa" not in df.columns or "capman_pa" not in df.columns:
        print(
            "[warn] Cannot detect endpoint: missing pirani_pa or capman_pa columns. "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        if "timestamp" in df.columns and len(df) > 0:
            return df["timestamp"].iloc[-1]
        return None
    
    # Ensure we have valid data
    valid_mask = df["pirani_pa"].notna() & df["capman_pa"].notna() & (df["capman_pa"] > 0)
    
    if not valid_mask.any():
        print(
            "[warn] No valid Pirani/Capman pressure readings for endpoint detection. "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        if "timestamp" in df.columns and len(df) > 0:
            return df["timestamp"].iloc[-1]
        return None
    
    # Calculate convergence condition
    diff_abs = (df["pirani_pa"] - df["capman_pa"]).abs()
    diff_rel = diff_abs / df["capman_pa"]
    
    # Convergence: either absolute OR relative tolerance is met
    converged = (diff_abs < abs_tol_pa) | (diff_rel < rel_tol)
    
    # Only consider valid rows
    converged = converged & valid_mask
    
    if not converged.any():
        print(
            "[warn] No convergence detected between Pirani and Capman pressures. "
            "Using last timestamp of primary phase as endpoint.",
            file=sys.stderr,
        )
        if "timestamp" in df.columns and len(df) > 0:
            return df["timestamp"].iloc[-1]
        return None
    
    # Find sustained convergence periods
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    converged_series = converged.reindex(df_sorted.index, fill_value=False)
    
    # Convert sustain_minutes to number of samples based on median sampling interval
    timestamps = df_sorted["timestamp"].dropna()
    if len(timestamps) < 2:
        print(
            "[warn] Insufficient timestamps for endpoint detection. "
            "Using last timestamp as endpoint.",
            file=sys.stderr,
        )
        return timestamps.iloc[-1] if len(timestamps) > 0 else None
    
    time_diffs = timestamps.diff().dt.total_seconds().dropna()
    median_interval_sec = time_diffs.median()
    
    if median_interval_sec <= 0:
        median_interval_sec = 60.0  # Default to 1 minute if calculation fails
    
    sustain_seconds = sustain_minutes * 60.0
    min_consecutive_points = int(np.ceil(sustain_seconds / median_interval_sec))
    min_consecutive_points = max(min_consecutive_points, 1)  # At least 1 point
    
    # Find runs of consecutive True values
    converged_values = converged_series.values
    indices = np.where(converged_values)[0]
    
    if len(indices) == 0:
        print(
            "[warn] No convergence points found. Using last timestamp as endpoint.",
            file=sys.stderr,
        )
        return df_sorted["timestamp"].iloc[-1]
    
    # Group consecutive indices into runs
    runs = []
    current_run = [indices[0]]
    
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_run.append(indices[i])
        else:
            if len(current_run) >= min_consecutive_points:
                runs.append(current_run)
            current_run = [indices[i]]
    
    # Don't forget the last run
    if len(current_run) >= min_consecutive_points:
        runs.append(current_run)
    
    if not runs:
        print(
            f"[warn] No sustained convergence found (need {min_consecutive_points} consecutive points). "
            "Using last timestamp as endpoint.",
            file=sys.stderr,
        )
        return df_sorted["timestamp"].iloc[-1]
    
    # Return the timestamp at the START of the first sustained convergence period
    first_run_start_idx = runs[0][0]
    endpoint_timestamp = df_sorted.iloc[first_run_start_idx]["timestamp"]
    
    print(f"[info] Primary drying endpoint detected at {endpoint_timestamp}. "
          f"All data after this timestamp will be excluded from the fit.")
    
    return endpoint_timestamp


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

    # Filter out post-endpoint data BEFORE segmentation
    if endpoint_timestamp is not None:
        pre_endpoint_mask = primary_df["timestamp"] <= endpoint_timestamp
        post_endpoint_count = (~pre_endpoint_mask).sum()
        if post_endpoint_count > 0:
            print(f"[info] Excluding {post_endpoint_count} post-endpoint rows from segmentation.")
        primary_df = primary_df[pre_endpoint_mask].copy()
        
        if primary_df.empty:
            raise ValueError(
                f"All primary drying data is after the endpoint timestamp ({endpoint_timestamp}). "
                "Check your endpoint detection or Phase code."
            )

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

        if len(group) < min_segment_points:
            skipped.append(
                f"cycle={cycle_id}_step={step_id}: kept "
                f"{len(group)}/{raw_row_count} rows "
                f"(<{min_segment_points} valid rows)"
            )
            continue

        # Check if entire segment is post-endpoint (shouldn't happen after filtering, but safety check)
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
        if endpoint_timestamp is not None:
            end_mask = group["timestamp"] <= endpoint_timestamp
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
