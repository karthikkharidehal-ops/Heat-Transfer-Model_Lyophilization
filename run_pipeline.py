"""
End-to-end local driver.

raw historian CSV
-> cleaned data
-> segmented by Cycle and Step inside Primary Drying
-> joint Pikal parameter fit

Runs entirely on your machine.

Usage example:

python run_pipeline.py export.csv --primary-phase-code 6 --fill-volume-ul 6 --calib 6:1.8 12:2.9 20:4.1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ingest import (
    load_raw,
    build_timestamp,
    normalize_units,
    flag_primary_drying_end,
)

from geometry import PCRTubeGeometry
from pikal_model import DryingSegment, fit_parameters_joint


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


def parse_calib(calib_args):
    """
    Parse --calib arguments like:

        6:1.8 12:2.9 20:4.1

    into SI pairs:

        [(6e-9 m3, 1.8e-3 m), ...]
    """
    points = []

    for arg in calib_args:
        try:
            vol_ul_str, height_mm_str = arg.split(":", maxsplit=1)
            vol_ul = float(vol_ul_str)
            height_mm = float(height_mm_str)
        except Exception as exc:
            raise ValueError(
                f"Invalid --calib value {arg!r}. "
                "Expected format vol_ul:height_mm, e.g. 6:1.8"
            ) from exc

        if vol_ul <= 0:
            raise ValueError(
                f"Invalid --calib volume in {arg!r}. Volume must be positive."
            )

        if height_mm <= 0:
            raise ValueError(
                f"Invalid --calib height in {arg!r}. Height must be positive."
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


def build_segments(
    df,
    primary_phase_code,
    tube,
    fill_volume_m3,
    min_segment_points=5,
    hold_step_threshold=20,
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
    min_segment_points : int, default 5
        Minimum number of valid rows per segment
    hold_step_threshold : int, default 20
        Number of consecutive data points with same shelf_setpt_k value
        to classify as a hold step. Fewer than this indicates a ramp step.
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

        # Detect if this is a hold step or ramp step based on shelf_setpt_k stability
        # If shelf_setpt_k has more than threshold consecutive identical values, it's a hold step
        is_hold_step = True  # Default to hold step
        if "shelf_setpt_k" in group.columns:
            shelf_setpt = group["shelf_setpt_k"].values
            
            # Find the maximum run length of consecutive identical values
            # (accounting for floating point tolerance)
            max_consecutive = 1
            current_consecutive = 1
            for i in range(1, len(shelf_setpt)):
                if np.isclose(shelf_setpt[i], shelf_setpt[i-1], rtol=1e-5):
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1
            
            # If max consecutive identical values < threshold, it's a ramp step
            is_hold_step = (max_consecutive >= hold_step_threshold)

        t0 = group["timestamp"].iloc[0]

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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Raw historian CSV export",
    )

    parser.add_argument(
        "--primary-phase-code",
        required=True,
        help="Value in the Phase column that means Primary Drying, e.g. 6",
    )

    parser.add_argument(
        "--fill-volume-ul",
        type=float,
        required=True,
        help="Fill volume in microliters, e.g. 6",
    )

    parser.add_argument(
        "--calib",
        nargs="*",
        default=[],
        help=(
            "Empirical fill-height calibration points as vol_ul:height_mm. "
            "Example: 6:1.8 12:2.9 20:4.1"
        ),
    )

    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("fit_report.txt"),
        help="Output text report file",
    )

    parser.add_argument(
        "--use-min-temp",
        action="store_true",
        default=False,
        help=(
            "Use the minimum probe reading (prod_min_k) instead of the average "
            "(prod_avg_k) as the product temperature. This approximates the ice-front "
            "temperature when probes are fully immersed and exposed to different "
            "sections of the freezing front."
        ),
    )

    parser.add_argument(
        "--use-transient",
        action="store_true",
        default=False,
        help=(
            "Enable transient heat accumulation mode in the Pikal model. This accounts "
            "for non-steady conditions during multi-step ramps (4-5 ramp steps) where "
            "the steady-state assumption breaks down. Recommended for recipes with "
            "frequent shelf temperature or pressure changes."
        ),
    )

    parser.add_argument(
        "--hybrid-mode",
        action="store_true",
        default=False,
        help=(
            "Use hybrid modeling: detect Hold vs Ramp steps based on shelf setpoint stability. "
            "If shelf_setpt_k has >=20 consecutive identical values, it's a Hold step (steady-state). "
            "Otherwise it's a Ramp step (transient mode). This automatically applies the appropriate "
            "physics based on actual process behavior rather than step numbering. Overrides --use-transient."
        ),
    )

    args = parser.parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    if args.fill_volume_ul <= 0:
        raise SystemExit("--fill-volume-ul must be positive.")

    try:
        calibration_points = parse_calib(args.calib)
    except ValueError as exc:
        raise SystemExit(str(exc))

    # Load and clean data.
    df = load_raw(args.input_csv)
    df = build_timestamp(df)
    df = normalize_units(df)
    df = flag_primary_drying_end(df)

    # Choose model input columns.
    pressure_col = choose_pressure_column(df)
    product_temp_col = choose_product_temperature_column(df, use_min=args.use_min_temp)

    # Convert user inputs.
    phase_code = coerce_phase_code(df, args.primary_phase_code)
    fill_volume_m3 = args.fill_volume_ul * 1e-9

    # Use calibrated geometry so --calib is actually applied.
    tube = CalibratedPCRTubeGeometry(calibration_points=calibration_points)

    # Build primary-drying segments.
    segments = build_segments(
        df=df,
        primary_phase_code=phase_code,
        tube=tube,
        fill_volume_m3=fill_volume_m3,
        hold_step_threshold=20,  # Detect hold vs ramp based on 20 consecutive identical shelf_setpt values
    )

    print(f"Built {len(segments)} segment(s): {[s.label for s in segments]}")
    
    # Report detected step types
    if args.hybrid_mode:
        hold_steps = [s.label for s in segments if s.is_hold_step]
        ramp_steps = [s.label for s in segments if not s.is_hold_step]
        print(f"[info] Hybrid mode: {len(hold_steps)} Hold step(s), {len(ramp_steps)} Ramp step(s)")
        if hold_steps:
            print(f"  Hold steps (steady-state): {hold_steps}")
        if ramp_steps:
            print(f"  Ramp steps (transient): {ramp_steps}")

    if len(segments) < 2:
        print(
            "[warn] Only one segment found -- Kv/Rp will NOT be uniquely "
            "identifiable from this alone. If possible, fit across multiple "
            "Steps/cycles with different shelf-temp or pressure setpoints.",
            file=sys.stderr,
        )

    # Run joint fit.
    try:
        # Hybrid mode overrides use_transient if enabled
        use_hybrid = args.hybrid_mode
        use_transient = args.use_transient or use_hybrid
        
        fitted = fit_parameters_joint(
            segments, 
            use_transient=use_transient,
            use_hybrid=use_hybrid
        )
    except Exception as exc:
        raise SystemExit(f"[error] Joint fit failed: {exc}")

    # Build report.
    lines = [
        "=== Joint Pikal model fit ===",
        f"Input file: {args.input_csv.name}",
        f"Primary phase code used: {phase_code}",
        f"Fill volume: {args.fill_volume_ul} uL",
        f"Pressure column used: {pressure_col}",
        f"Product temperature column used: {product_temp_col}",
        f"Using minimum probe temperature (ice-front approx): {args.use_min_temp}",
        f"Using transient mode (for multi-step ramps): {use_transient and not use_hybrid}",
        f"Using hybrid mode (shelf-setpt based Hold/Ramp detection): {use_hybrid}",
        f"Calibration points used: {len(calibration_points)}",
        f"Segments used: {[s.label for s in segments]}",
        "",
        "Fitted parameters:",
        f"  Kv  = {fitted['Kv']:.6f}",
        f"  Rp0 = {fitted['Rp0']:.6f}",
        f"  A1  = {fitted['A1']:.6f}",
        f"  A2  = {fitted['A2']:.6f}",
        f"  Overall RMS error: {fitted['rms_error_k']:.6f} K",
        f"  Converged: {fitted['converged']}",
        "",
        "Per-segment RMS (K) -- large outliers here suggest a mislabeled",
        "phase/step or a segment that does not belong in the joint fit:",
    ]

    for label, rms in fitted.get("per_segment_rms_k", {}).items():
        lines.append(f"  {label}: {rms:.6f}")

    report = "\n".join(lines)

    print("\n" + report)

    args.report_out.write_text(report, encoding="utf-8")
    print(f"\nSaved report -> {args.report_out}")


if __name__ == "__main__":
    main()