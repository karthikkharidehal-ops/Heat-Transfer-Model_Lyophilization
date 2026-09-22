"""
Contact Area Sensitivity Analysis for Lyophilization Digital Twin

This script performs a sensitivity analysis on the contact_efficiency parameter
to determine the optimal value that minimizes the Step 4 (Hold, steady-state) RMS error.

The contact area efficiency factor (currently 22% from Graberg's thesis) may not be 
correct for this specific tube-and-block configuration. This analysis helps identify
the optimal contact efficiency by testing multiple values and comparing fit quality.

Event: E-DIAG-CONTACT-001
Phase: Phase 0 / Phase 1 blocked
Gate impact: BLOCKING
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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

from pikal_model import fit_parameters_joint


# Contact efficiency values to test
CONTACT_EFFICIENCY_VALUES = [0.05, 0.10, 0.15, 0.22, 0.30, 0.50, 0.75, 1.00]


def run_sensitivity_analysis(
    input_csv: Path,
    primary_phase_code,
    fill_volume_ul: float,
    calib_points: list,
    use_min_temp: bool = False,
    use_transient: bool = False,
    use_hybrid: bool = False,
):
    """
    Run sensitivity analysis across contact_efficiency values.
    
    Returns a DataFrame with results for each contact_efficiency value.
    """
    # Load and clean data
    df = load_raw(input_csv)
    df = build_timestamp(df)
    df = normalize_units(df)
    df = flag_primary_drying_end(df)
    
    # Choose model input columns
    pressure_col = choose_pressure_column(df)
    product_temp_col = choose_product_temperature_column(df, use_min=use_min_temp)
    
    # This writes df['product_temp_k'] which is required by build_segments
    
    # Convert user inputs
    phase_code = coerce_phase_code(df, primary_phase_code)
    fill_volume_m3 = fill_volume_ul * 1e-9
    
    # Detect endpoint once (independent of contact_efficiency)
    endpoint_ts, endpoint_diagnostics = find_endpoint(
        df,
        rolling_window_minutes=30.0,
        sustain_minutes=30.0,
        min_baseline_fraction=0.25,
    )
    
    results = []
    
    for contact_eff in CONTACT_EFFICIENCY_VALUES:
        print(f"\nTesting contact_efficiency = {contact_eff:.2f}...")
        
        # Create geometry with current contact efficiency
        tube = CalibratedPCRTubeGeometry(
            calibration_points=calib_points,
            contact_efficiency=contact_eff,
        )
        
        # Build segments
        segments = build_segments(
            df=df,
            primary_phase_code=phase_code,
            tube=tube,
            fill_volume_m3=fill_volume_m3,
            endpoint_timestamp=endpoint_ts,
            shelf_temp_std_threshold=0.1,
            pressure_std_threshold=5.0,
        )
        
        if len(segments) < 2:
            print(f"  [warn] Only {len(segments)} segment(s) found; skipping fit.")
            results.append({
                'contact_efficiency': contact_eff,
                'Kv': np.nan,
                'Rp0': np.nan,
                'A1': np.nan,
                'A2': np.nan,
                'overall_rms_k': np.nan,
                'step_4_rms_k': np.nan,
                'total_htc': np.nan,  # Kv * contact_area
                'error': 'Insufficient segments',
            })
            continue
        
        # Run joint fit
        try:
            fitted = fit_parameters_joint(
                segments,
                use_transient=use_transient,
                use_hybrid=use_hybrid,
            )
        except Exception as exc:
            print(f"  [error] Fit failed: {exc}")
            results.append({
                'contact_efficiency': contact_eff,
                'Kv': np.nan,
                'Rp0': np.nan,
                'A1': np.nan,
                'A2': np.nan,
                'overall_rms_k': np.nan,
                'step_4_rms_k': np.nan,
                'total_htc': np.nan,
                'error': str(exc),
            })
            continue
        
        # Extract fitted parameters
        Kv = fitted['Kv']
        Rp0 = fitted['Rp0']
        A1 = fitted['A1']
        A2 = fitted['A2']
        overall_rms = fitted['rms_error_k']
        per_segment_rms = fitted.get('per_segment_rms_k', {})
        
        # Find Step 4 RMS error (or any Hold step)
        step_4_rms = np.nan
        for label, rms in per_segment_rms.items():
            if 'step=4' in label.lower() or 'step_4' in label.lower():
                step_4_rms = rms
                break
        
        # If no explicit Step 4, use the first Hold step
        if np.isnan(step_4_rms):
            for seg in segments:
                if seg.is_hold_step:
                    label = seg.label
                    if label in per_segment_rms:
                        step_4_rms = per_segment_rms[label]
                        break
        
        # Calculate total heat transfer coefficient: Kv * contact_area
        # Contact area is based on tube geometry and contact_efficiency
        contact_area = tube.outer_radius ** 2 * np.pi * contact_eff
        total_htc = Kv * contact_area
        
        results.append({
            'contact_efficiency': contact_eff,
            'Kv': Kv,
            'Rp0': Rp0,
            'A1': A1,
            'A2': A2,
            'overall_rms_k': overall_rms,
            'step_4_rms_k': step_4_rms,
            'total_htc': total_htc,
            'error': None,
        })
        
        print(f"  Kv={Kv:.6f}, Rp0={Rp0:.6f}, A1={A1:.6f}, A2={A2:.6f}")
        print(f"  Overall RMS: {overall_rms:.4f} K, Step 4 RMS: {step_4_rms:.4f} K")
        print(f"  Total HTC (Kv * contact_area): {total_htc:.6f} W/K")
    
    return pd.DataFrame(results), endpoint_diagnostics


def print_summary_table(results_df: pd.DataFrame):
    """Print a formatted summary table of results."""
    print("\n" + "=" * 100)
    print("CONTACT EFFICIENCY SENSITIVITY ANALYSIS RESULTS")
    print("=" * 100)
    
    # Header
    header = (
        f"{'Contact Eff.':>14} | {'Kv':>12} | {'Rp0':>10} | {'A1':>10} | {'A2':>10} | "
        f"{'Overall RMS':>12} | {'Step 4 RMS':>12} | {'Total HTC':>12}"
    )
    print(header)
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        if row['error']:
            print(f"{row['contact_efficiency']:>14.2f} | FIT ERROR: {row['error'][:50]}")
        else:
            line = (
                f"{row['contact_efficiency']:>14.2f} | {row['Kv']:>12.6f} | {row['Rp0']:>10.6f} | "
                f"{row['A1']:>10.6f} | {row['A2']:>10.6f} | {row['overall_rms_k']:>12.4f} | "
                f"{row['step_4_rms_k']:>12.4f} | {row['total_htc']:>12.6f}"
            )
            print(line)
    
    print("=" * 100)


def find_optimal_contact_efficiency(results_df: pd.DataFrame):
    """Find the contact_efficiency that minimizes Step 4 RMS error."""
    # Filter out rows with errors
    valid_df = results_df[results_df['error'].isna()].copy()
    
    if valid_df.empty:
        print("\n[ERROR] No valid results to analyze.")
        return None
    
    # Check if we have any Step 4 RMS values
    if valid_df['step_4_rms_k'].isna().all():
        print("\n[WARN] No Step 4 RMS values available; using overall RMS instead.")
        metric_col = 'overall_rms_k'
        metric_name = 'Overall RMS'
    else:
        metric_col = 'step_4_rms_k'
        metric_name = 'Step 4 RMS'
    
    # Find minimum
    min_idx = valid_df[metric_col].idxmin()
    optimal_row = valid_df.loc[min_idx]
    
    print(f"\n*** OPTIMAL CONTACT EFFICIENCY: {optimal_row['contact_efficiency']:.2f} ***")
    print(f"    Minimizes {metric_name}: {optimal_row[metric_col]:.4f} K")
    print(f"    Corresponding Kv: {optimal_row['Kv']:.6f} W/m²/K")
    print(f"    Total HTC: {optimal_row['total_htc']:.6f} W/K")
    
    return optimal_row


def generate_plot(results_df: pd.DataFrame, output_path: Path, optimal_value: float):
    """Generate the sensitivity analysis plot."""
    valid_df = results_df[results_df['error'].isna()].copy()
    
    if valid_df.empty:
        print("[WARN] No valid results to plot.")
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # X-axis: contact_efficiency (log scale)
    x = valid_df['contact_efficiency'].values
    
    # Left Y-axis: Step 4 RMS error (or overall RMS if Step 4 not available)
    if valid_df['step_4_rms_k'].isna().all():
        y1 = valid_df['overall_rms_k'].values
        ylabel1 = 'Overall RMS Error (K)'
    else:
        y1 = valid_df['step_4_rms_k'].values
        ylabel1 = 'Step 4 RMS Error (K)'
    
    # Right Y-axis: Fitted Kv
    y2 = valid_df['Kv'].values
    
    # Plot Step 4 RMS on left axis
    color1 = 'tab:red'
    ax1.set_xlabel('Contact Efficiency')
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.semilogx(x, y1, 'o-', color=color1, linewidth=2, markersize=8, label='RMS Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot Kv on right axis
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Fitted Kv (W/m²/K)', color=color2)
    ax2.semilogx(x, y2, 's--', color=color2, linewidth=2, markersize=8, label='Kv')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Mark optimal value with vertical line
    if optimal_value is not None:
        ax1.axvline(x=optimal_value, color='green', linestyle='--', linewidth=2, 
                   label=f'Optimal (eff={optimal_value:.2f})')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('Contact Efficiency Sensitivity Analysis')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot -> {output_path}")
    plt.close(fig)


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
        help="Empirical fill-height calibration points as vol_ul:height_mm",
    )
    
    parser.add_argument(
        "--use-min-temp",
        action="store_true",
        default=False,
        help="Use minimum probe reading (prod_min_k) as product temperature",
    )
    
    parser.add_argument(
        "--use-transient",
        action="store_true",
        default=False,
        help="Enable transient heat accumulation mode",
    )
    
    parser.add_argument(
        "--hybrid-mode",
        action="store_true",
        default=False,
        help="Use hybrid modeling with Hold/Ramp detection based on measured variance",
    )
    
    parser.add_argument(
        "--results-out",
        type=Path,
        default=Path("sensitivity_contact_area_results.csv"),
        help="Output CSV file for results",
    )
    
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("sensitivity_contact_area.png"),
        help="Output PNG file for plot",
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
    
    print("=" * 80)
    print("CONTACT AREA SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Input file: {args.input_csv}")
    print(f"Primary phase code: {args.primary_phase_code}")
    print(f"Fill volume: {args.fill_volume_ul} uL")
    print(f"Calibration points: {len(calibration_points)}")
    print(f"Contact efficiency values to test: {CONTACT_EFFICIENCY_VALUES}")
    print("=" * 80)
    
    # Run sensitivity analysis
    results_df, endpoint_diagnostics = run_sensitivity_analysis(
        input_csv=args.input_csv,
        primary_phase_code=args.primary_phase_code,
        fill_volume_ul=args.fill_volume_ul,
        calib_points=calibration_points,
        use_min_temp=args.use_min_temp,
        use_transient=args.use_transient,
        use_hybrid=args.hybrid_mode,
    )
    
    # Print summary table
    print_summary_table(results_df)
    
    # Find optimal contact efficiency
    optimal_row = find_optimal_contact_efficiency(results_df)
    optimal_value = optimal_row['contact_efficiency'] if optimal_row is not None else None
    
    # Save results to CSV
    results_df.to_csv(args.results_out, index=False)
    print(f"\nSaved results -> {args.results_out}")
    
    # Generate plot
    generate_plot(results_df, args.plot_out, optimal_value)
    
    # Report endpoint diagnostics
    print("\n" + "-" * 80)
    print("ENDPOINT DETECTION DIAGNOSTICS:")
    print(f"  Primary baseline: {endpoint_diagnostics.get('primary_baseline', 'N/A')} Pa")
    print(f"  Post baseline: {endpoint_diagnostics.get('post_baseline', 'N/A')} Pa")
    print(f"  Threshold: {endpoint_diagnostics.get('threshold', 'N/A')} Pa")
    print(f"  Detected: {endpoint_diagnostics.get('detected', 'N/A')}")
    print("-" * 80)
    
    # Final recommendation
    if optimal_value is not None:
        print(f"\n*** RECOMMENDATION: Use contact_efficiency = {optimal_value:.2f} ***")
        print("    This value minimizes the Step 4 (Hold step) RMS error.")
        print("    Update your pipeline runs with --contact-efficiency {:.2f}".format(optimal_value))


if __name__ == "__main__":
    main()
