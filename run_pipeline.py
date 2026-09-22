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

from core_pipeline import (
    CalibratedPCRTubeGeometry,
    build_segments,
    choose_pressure_column,
    choose_product_temperature_column,
    coerce_phase_code,
    parse_calib,
    find_endpoint,
)

from pikal_model import fit_parameters_joint, simulate_cycle, simulate_continuous_primary_drying


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
        "--contact-efficiency",
        type=float,
        default=0.22,
        help=(
            "Contact efficiency factor for PCR tube thermal contact with aluminum block/shelf "
            "(default 0.22 per Graberg thesis)"
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
    tube = CalibratedPCRTubeGeometry(
        calibration_points=calibration_points,
        contact_efficiency=args.contact_efficiency
    )

    # Detect primary drying endpoint using Pirani/Capman convergence
    endpoint_ts = find_endpoint(
        df, 
        abs_tol_pa=1.0, 
        rel_tol=0.15, 
        sustain_minutes=20.0
    )
    
    # Calculate endpoint as time from start of process (in seconds)
    first_timestamp = df["timestamp"].min()
    endpoint_time_from_start = None
    if endpoint_ts is not None and pd.notna(first_timestamp):
        endpoint_time_from_start = (endpoint_ts - first_timestamp).total_seconds()

    # Build primary-drying segments (truncated at endpoint)
    segments = build_segments(
        df=df,
        primary_phase_code=phase_code,
        tube=tube,
        fill_volume_m3=fill_volume_m3,
        endpoint_timestamp=endpoint_ts,
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
        f"Primary drying endpoint detected at {endpoint_ts}. All data after this timestamp was excluded from the fit.",
        f"Pirani and Capacitance Manometer (CM) merge time: {endpoint_ts}",
    ]
    
    if endpoint_time_from_start is not None:
        lines.append(f"Endpoint time from start of process: {endpoint_time_from_start:.1f} seconds ({endpoint_time_from_start/60:.1f} minutes)")
    
    lines.extend([
        "(This marks the primary drying endpoint when water vapor sublimation ceased)",
        "",
        "Steady-state detection based on measured signal variance:",
        f"  Shelf temp threshold: std < 0.1 K",
        f"  Pressure threshold: std < 5.0 Pa (capacitance manometer)",
        "",
        "Detected step classification (Hold vs Ramp):",
    ])
    
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
    
    # Add endpoint truncation diagnostics
    lines.extend([
        "",
        "Endpoint truncation diagnostics:",
    ])
    
    # Re-run build_segments to capture truncation diagnostics
    import io as io_module
    from contextlib import redirect_stdout
    
    diag_output = io_module.StringIO()
    with redirect_stdout(diag_output):
        build_segments(
            df_clean,
            phase_code,
            tube,
            fill_volume_m3,
            endpoint_timestamp=endpoint_ts,
            min_segment_points=args.min_segment_points,
        )
    truncation_diagnostics = diag_output.getvalue()
    
    if truncation_diagnostics.strip():
        for line in truncation_diagnostics.strip().split('\n'):
            if line.startswith('[DIAG]'):
                lines.append(line)
    else:
        lines.append("  No rows dropped due to endpoint truncation.")
    
    # Count total rows excluded
    total_rows_before = len(df_clean[df_clean["Phase"] == phase_code])
    total_rows_after = sum(len(seg.t_s) for seg in segments)
    rows_excluded = total_rows_before - total_rows_after
    
    lines.extend([
        "",
        f"Data exclusion summary:",
        f"  Total rows in primary phase before truncation: {total_rows_before}",
        f"  Total rows used in fit after truncation: {total_rows_after}",
        f"  Total rows excluded (endpoint + invalid): {rows_excluded}",
    ])
    
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

    # Add state continuity diagnostics if using continuous simulation
    if use_transient or use_hybrid:
        lines.extend([
            "",
            "[STATE-CONTINUITY] Diagnostics (from continuous simulation):",
        ])
        # Re-run simulation to capture state continuity prints
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            simulate_continuous_primary_drying(
                segments, Kv, Rp0, A1, A2, use_hybrid=use_hybrid
            )
        state_output = f.getvalue()
        lines.append(state_output)
    
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