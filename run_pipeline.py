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
from pikal_model import DryingSegment, fit_parameters_joint, simulate_cycle, simulate_continuous_primary_drying


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
            "Use hybrid modeling: detect Hold vs Ramp steps based on MEASURED signal variance. "
            "A step is classified as Hold (steady-state) ONLY if std(shelf_temp_k) < 0.1 K AND "
            "std(capman_pa) < 5.0 Pa. Otherwise it's a Ramp step (transient mode). This uses "
            "actual physical measurements rather than PLC setpoints. Overrides --use-transient."
        ),
    )

    parser.add_argument(
        "--plot-residuals",
        action="store_true",
        default=False,
        help=(
            "Generate residual plots showing measured vs simulated product temperature "
            "and residuals for each segment. Saves plots as PNG files."
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
        shelf_temp_std_threshold=0.1,  # Hold step if std(shelf_temp_k) < 0.1 K
        pressure_std_threshold=5.0,   # AND std(capman_pa) < 5.0 Pa
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

    # Extract fitted parameters for simulation
    Kv, Rp0, A1, A2 = fitted["Kv"], fitted["Rp0"], fitted["A1"], fitted["A2"]
    
    # Build report with step-type aware per-segment data
    lines = [
        "=== Joint Pikal model fit ===",
        f"Input file: {args.input_csv.name}",
        f"Primary phase code used: {phase_code}",
        f"Fill volume: {args.fill_volume_ul} uL",
        f"Pressure column used: {pressure_col}",
        f"Product temperature column used: {product_temp_col}",
        f"Using minimum probe temperature (ice-front approx): {args.use_min_temp}",
        f"Using transient mode (for multi-step ramps): {use_transient and not use_hybrid}",
        f"Using hybrid mode (measured signal variance Hold/Ramp detection): {use_hybrid}",
        f"Calibration points used: {len(calibration_points)}",
        "",
        "Steady-state detection based on measured signal variance:",
        f"  Shelf temp threshold: std < 0.1 K",
        f"  Pressure threshold: std < 5.0 Pa (capacitance manometer)",
        "",
        "Detected step classification (Hold vs Ramp):",
    ]
    
    # Report detected step types with their characteristics AND measured variance proof
    for seg in segments:
        step_type = "Hold (steady-state)" if seg.is_hold_step else "Ramp (transient)"
        # Get shelf temp range for this segment
        Ts_min = seg.Ts_k.min() - 273.15  # Convert to Celsius
        Ts_max = seg.Ts_k.max() - 273.15
        Pc_mean = seg.Pc_pa.mean() / 100.0  # Convert to hPa/mbar
        
        lines.append(f"  {seg.label}: {step_type}")
        lines.append(f"    Shelf temp: {Ts_min:.1f}C to {Ts_max:.1f}C")
        lines.append(f"    Chamber pressure: {Pc_mean:.2f} mbar")
        lines.append(f"    Duration: {seg.t_s[-1]:.0f}s ({seg.t_s[-1]/60:.1f} min)")
        lines.append(f"    Measured steady-state variance proof:")
        lines.append(f"      std(shelf_temp_k) = {seg._shelf_temp_std:.4f} K {'< 0.1 K ✓' if seg._shelf_temp_std < 0.1 else '>= 0.1 K ✗'}")
        lines.append(f"      std(capman_pa) = {seg._pressure_std:.4f} Pa {'< 5.0 Pa ✓' if seg._pressure_std < 5.0 else '>= 5.0 Pa ✗'}")
        classification_reason = "BOTH stable → HOLD" if seg.is_hold_step else "One or both signals unstable → RAMP"
        lines.append(f"    Classification: {classification_reason}")
    
    lines.extend([
        "",
        "Fitted parameters:",
        f"  Kv  = {fitted['Kv']:.6f}",
        f"  Rp0 = {fitted['Rp0']:.6f}",
        f"  A1  = {fitted['A1']:.6f}",
        f"  A2  = {fitted['A2']:.6f}",
        f"  Overall RMS error: {fitted['rms_error_k']:.6f} K",
        f"  Converged: {fitted['converged']}",
        "",
        "Per-segment fit quality (RMS error in K):",
    ])

    for label, rms in fitted.get("per_segment_rms_k", {}).items():
        lines.append(f"  {label}: {rms:.6f} K")

    report = "\n".join(lines)

    print("\n" + report)

    args.report_out.write_text(report, encoding="utf-8")
    print(f"\nSaved report -> {args.report_out}")
    
    # Generate residual plots if requested
    if args.plot_residuals:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            print("\nGenerating residual plots...")
            
            # Use continuous simulation across all segments when appropriate
            if use_transient or use_hybrid:
                all_Tp_sim = simulate_continuous_primary_drying(
                    segments, Kv, Rp0, A1, A2, use_hybrid=use_hybrid
                )
            else:
                # Legacy mode: independent simulations
                all_Tp_sim = []
                for seg in segments:
                    Tp_sim = simulate_cycle(
                        seg.t_s, seg.Ts_k, seg.Pc_pa, seg.tube,
                        seg.fill_volume_m3, Kv, Rp0, A1, A2, use_transient=False
                    )
                    all_Tp_sim.append(Tp_sim)
            
            for i, seg in enumerate(segments):
                Tp_sim = all_Tp_sim[i]
                
                # Calculate residuals
                residuals = Tp_sim - seg.Tp_measured_k
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                
                # Top panel: Measured vs Simulated
                ax1.plot(seg.t_s / 60, seg.Tp_measured_k - 273.15, 'b-', label='Measured', linewidth=1.5)
                ax1.plot(seg.t_s / 60, Tp_sim - 273.15, 'r--', label='Simulated', linewidth=1.5)
                ax1.set_ylabel('Temperature (°C)')
                step_type = "Hold (steady-state)" if seg.is_hold_step else "Ramp (transient)"
                continuity_note = " [continuous]" if (use_transient or use_hybrid) else ""
                variance_proof = f"σ(Ts)={seg._shelf_temp_std:.3f}K, σ(P)={seg._pressure_std:.2f}Pa"
                ax1.set_title(f'{seg.label} - Step Type: {step_type}{continuity_note}\n{variance_proof}')
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                
                # Bottom panel: Residuals
                ax2.plot(seg.t_s / 60, residuals, 'g-', linewidth=1)
                ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax2.fill_between(seg.t_s / 60, residuals, 0, alpha=0.3, color='green')
                ax2.set_xlabel('Time (min)')
                ax2.set_ylabel('Residual (°C)')
                ax2.set_title(f'Residuals (Simulated - Measured), RMS = {np.sqrt(np.mean(residuals**2)):.3f} K')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                plot_filename = f"residual_{seg.label.replace('=', '_').replace('cycle_', 'c').replace('step_', 's')}.png"
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"  Saved: {plot_filename}")
                plt.close(fig)
            
            print("Residual plots generated successfully.")
        except ImportError:
            print("[warn] matplotlib not installed; cannot generate residual plots. Install with: pip install matplotlib")
        except Exception as exc:
            print(f"[warn] Failed to generate residual plots: {exc}")


if __name__ == "__main__":
    main()