"""
Web-based GUI for Lyophilization Pikal Model Pipeline

Run with: python gui_web.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import io
import threading
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file

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
from report_builders import build_mass_balance_lines

# --- Single measured anchor (event E-GEOM-ANCHOR-001) -----------------------
# Anchored geometry is the DEFAULT and ONLY trusted mode for this batch:
#   16.0 uL  <->  4.0 mm fill height in production PCR tubes.
# The area profile A(h) is an explicit, selectable assumption:
#   cone_tip | scaled_frustum   (default: cone_tip)
# Multi-point calibration (--calib / "Calibration Points") is DEFERRED
# (open assumption A-GEOM-004): it is ignored in anchored mode with a loud
# warning, and is MANDATORY before any recipe optimization or advisory MPC.
from geometry import (
    ANCHOR_FILL_VOLUME_UL,
    ANCHOR_FILL_HEIGHT_MM,
    GEOMETRY_PROFILES,
    AnchoredTubeGeometry,
    make_anchored_tube,
    build_geometry_table_rows,
)

# Sensitivity sweep values shared with sensitivity_contact_area.py
try:
    from sensitivity_contact_area import CONTACT_EFFICIENCY_VALUES
except Exception:  # pragma: no cover - optional dependency path
    CONTACT_EFFICIENCY_VALUES = [0.05, 0.10, 0.15, 0.22, 0.30, 0.50, 0.75, 1.00]


def _anchored_sensitivity_fit(
    input_path,
    primary_phase_code,
    fill_volume_ul: float,
    contact_efficiencies,
    geometry_profile: str,
    use_min_temp: bool,
    use_transient: bool,
    use_hybrid: bool,
):
    """Contact-efficiency sensitivity sweep using the ANCHORED geometry.

    Mirrors sensitivity_contact_area.run_sensitivity_analysis() but rebuilds
    an AnchoredTubeGeometry per contact efficiency (the legacy module still
    uses CalibratedPCRTubeGeometry, which contradicts E-GEOM-ANCHOR-001).
    Returns a pandas DataFrame of results.
    """
    df = load_raw(input_path)
    df = build_timestamp(df)
    df = normalize_units(df)
    df = flag_primary_drying_end(df)

    pressure_col = choose_pressure_column(df)
    product_temp_col = choose_product_temperature_column(df, use_min=use_min_temp)
    phase_code = coerce_phase_code(df, primary_phase_code)
    fill_volume_m3 = fill_volume_ul * 1e-9

    endpoint_result = find_endpoint(
        df, rolling_window_minutes=30.0, sustain_minutes=30.0, min_baseline_fraction=0.25
    )
    endpoint_ts = endpoint_result[0] if isinstance(endpoint_result, tuple) else endpoint_result

    rows = []
    for contact_eff in contact_efficiencies:
        tube = make_anchored_tube(
            profile=geometry_profile,
            fill_volume_m3=fill_volume_m3,
            contact_efficiency=contact_eff,
        )
        try:
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
                raise RuntimeError(f"only {len(segments)} segment(s) found")
            fitted = fit_parameters_joint(
                segments,
                use_transient=use_transient,
                use_hybrid=use_hybrid,
            )
            step4_rms = np.nan
            for label, rms in fitted.get("per_segment_rms_k", {}).items():
                if "step=4" in label or "Step 4" in label or label.endswith("4"):
                    step4_rms = rms
                    break
            rows.append({
                "contact_efficiency": contact_eff,
                "Kv": fitted["Kv"],
                "Rp0": fitted["Rp0"],
                "A1": fitted["A1"],
                "A2": fitted["A2"],
                "overall_rms_k": fitted["rms_error_k"],
                "step_4_rms_k": step4_rms,
                "total_htc": fitted["Kv"] * tube.contact_area_m2(tube.fill_height(fill_volume_m3)),
                "error": None,
            })
        except Exception as exc:  # keep sweeping; record per-row failure
            rows.append({
                "contact_efficiency": contact_eff,
                "Kv": np.nan, "Rp0": np.nan, "A1": np.nan, "A2": np.nan,
                "overall_rms_k": np.nan, "step_4_rms_k": np.nan,
                "total_htc": np.nan, "error": str(exc),
            })
    return pd.DataFrame(rows)


app = Flask(__name__)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lyophilization Pikal Model Pipeline</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5;
        }
        .container { 
            max-width: 1000px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #2c3e50; 
            margin-bottom: 10px;
            text-align: center;
        }
        .subtitle {
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 25px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .section h2 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.2em;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #34495e;
        }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        input[type="file"] {
            padding: 10px 0;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .checkbox-group input {
            margin-right: 10px;
            width: 18px;
            height: 18px;
        }
        .checkbox-group label {
            margin: 0;
            font-weight: normal;
        }
        .help-text {
            color: #7f8c8d;
            font-size: 0.85em;
            margin-top: 5px;
        }
        .btn {
            background: #27ae60;
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .btn:hover { background: #219a52; }
        .btn:disabled { background: #95a5a6; cursor: not-allowed; }
        .btn-container { text-align: center; margin-top: 30px; }
        #log {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            margin-top: 20px;
            display: none;
        }
        .log-error { color: #e74c3c; }
        .log-warning { color: #f39c12; }
        .log-success { color: #2ecc71; }
        .log-info { color: #3498db; }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .status-bar {
            padding: 10px;
            margin-top: 20px;
            border-radius: 5px;
            text-align: center;
            display: none;
        }
        .status-running { background: #fff3cd; color: #856404; }
        .status-success { background: #d4edda; color: #155724; }
        .status-error { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Lyophilization Pikal Model Pipeline</h1>
        <p class="subtitle">Fit Kv, Rp0, A1, A2 parameters from lyophilization cycle data</p>
        
        <form id="pipeline-form">
            <div class="section">
                <h2>📁 Input/Output Files</h2>
                <div class="form-group">
                    <label for="input_csv">Input CSV File:</label>
                    <input type="file" id="input_csv" name="input_csv" accept=".csv" required>
                </div>
                <div class="form-group">
                    <label for="report_out">Report Output Filename:</label>
                    <input type="text" id="report_out" name="report_out" value="fit_report.txt">
                </div>
            </div>
            
            <div class="section">
                <h2>&#128207; Geometry (single measured anchor — E-GEOM-ANCHOR-001)</h2>
                <div style="margin-bottom: 15px; padding: 10px; background: #fff8e1; border-left: 4px solid #f39c12; border-radius: 4px;">
                    <strong>Anchored mode is the only trusted geometry for this batch.</strong><br>
                    Measured anchor: <strong>16.0 &#xB5;L &#8596; 4.0 mm fill height</strong> (production PCR tubes).<br>
                    Fill height is FIXED at 4.0 mm; multi-point calibration is <strong>DEFERRED</strong>
                    (open assumption A-GEOM-004) and is <strong>mandatory before any recipe
                    optimization or advisory MPC use</strong>.
                </div>
                <div class="form-group">
                    <label for="geometry_profile">Area profile (explicit assumption):</label>
                    <select id="geometry_profile" name="geometry_profile">
                        <option value="cone_tip" selected>cone_tip (default) &#8212; r(h) = k&#183;h, V(H) = (&#960;/3)k&#178;H&#179;</option>
                        <option value="scaled_frustum">scaled_frustum &#8212; A(h) = c&#183;A_ideal(h), V(H) = c&#183;V_ideal(H)</option>
                    </select>
                    <div class="help-text">CLI equivalent: --geometry-profile {cone_tip, scaled_frustum}. Both profiles are pinned so V(4.0 mm) = anchor volume exactly.</div>
                </div>
                <div class="form-group">
                    <label for="contact_efficiency">Contact efficiency:</label>
                    <input type="number" id="contact_efficiency" name="contact_efficiency" value="0.22" step="0.01" min="0.01" max="1.0">
                    <div class="help-text">Effective thermal contact fraction (default 0.22 per Graberg thesis)</div>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="anchored_mode" name="anchored_mode" checked disabled>
                    <label for="anchored_mode">Anchored geometry (16.0 &#xB5;L = 4.0 mm) &#8212; locked for this batch</label>
                </div>
                <div class="form-group">
                    <label for="calibration">Calibration Points (vol_ul:height_mm) &#8212; DEFERRED / IGNORED:</label>
                    <input type="text" id="calibration" name="calibration" placeholder="(deferred &#8212; leave empty)" autocomplete="off">
                    <div class="help-text" style="color:#c0392b;">&#9888; Multi-point calibration is deferred (A-GEOM-004). In anchored mode this field is IGNORED with a loud warning; do not rely on it until calibration is completed.</div>
                </div>
            </div>

            <div class="section">
                <h2>&#9881;&#65039; Model Parameters</h2>
                <div class="form-group">
                    <label for="phase_code">Primary Phase Code:</label>
                    <input type="text" id="phase_code" name="phase_code" value="6" placeholder="e.g., 6">
                    <div class="help-text">The value in the Phase column that indicates primary drying</div>
                </div>
                <div class="form-group">
                    <label for="fill_volume">Fill Volume (&#xB5;L):</label>
                    <input type="number" id="fill_volume" name="fill_volume" value="16" step="0.1" min="0">
                    <div class="help-text">Anchor: 16.0 &#xB5;L (= 4.0 mm). Fill HEIGHT stays fixed at 4.0 mm; this value scales the anchor volume for the active area profile.</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔬 Simulation Options</h2>
                <div class="checkbox-group">
                    <input type="checkbox" id="use_min_temp" name="use_min_temp">
                    <label for="use_min_temp">Use minimum probe temperature (ice-front approximation)</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="use_transient" name="use_transient">
                    <label for="use_transient">Transient mode (for multi-step ramps)</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="hybrid_mode" name="hybrid_mode" checked>
                    <label for="hybrid_mode">Hybrid mode (auto-detect Hold vs Ramp steps) ⭐ Recommended</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="plot_residuals" name="plot_residuals">
                    <label for="plot_residuals">Generate residual plots (PNG files)</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="run_sensitivity" name="run_sensitivity">
                    <label for="run_sensitivity">Run contact area sensitivity analysis (diagnose Step 4 error)</label>
                </div>
                
                <div style="margin-top: 15px; padding: 10px; background: #e8f4f8; border-radius: 4px;">
                    <strong>Endpoint Detection Parameters:</strong>
                    <div class="form-group" style="margin-top: 10px;">
                        <label for="rolling_window_minutes">Rolling Window (min):</label>
                        <input type="number" id="rolling_window_minutes" name="rolling_window_minutes" value="30.0" step="1.0" min="1.0">
                    </div>
                    <div class="form-group">
                        <label for="sustain_minutes">Sustain Duration (min):</label>
                        <input type="number" id="sustain_minutes" name="sustain_minutes" value="30.0" step="1.0" min="1.0">
                    </div>
                </div>
                
                <div class="help-text" style="margin-top: 15px; padding: 10px; background: #e8f4f8; border-radius: 4px;">
                    💡 <strong>Hybrid mode</strong> automatically detects Hold steps (steady shelf temp) vs Ramp steps 
                    (changing shelf temp) and applies appropriate physics. This is the recommended approach for most cycles.
                </div>
            </div>
            
            <div class="btn-container">
                <button type="submit" class="btn" id="run-btn">
                    🚀 Run Pipeline
                </button>
            </div>
        </form>
        
        <div id="status-bar" class="status-bar"></div>
        <div id="log"></div>
    </div>
    
    <script>
        document.getElementById('pipeline-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const runBtn = document.getElementById('run-btn');
            const logDiv = document.getElementById('log');
            const statusBar = document.getElementById('status-bar');
            
            runBtn.disabled = true;
            runBtn.innerHTML = '<span class="spinner"></span> Running...';
            logDiv.style.display = 'block';
            logDiv.innerHTML = '';
            statusBar.style.display = 'none';
            
            const formData = new FormData(this);
            
            try {
                const response = await fetch('/run', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    logDiv.innerHTML = result.log.map(entry => 
                        `<span class="log-${entry.tag}">${entry.text}</span>`
                    ).join('\\n');
                    
                    statusBar.className = 'status-bar status-success';
                    statusBar.innerHTML = '✅ Pipeline completed successfully!';
                    statusBar.style.display = 'block';
                    
                    // Auto-download report
                    if (result.report_url) {
                        setTimeout(() => {
                            window.location.href = result.report_url;
                        }, 1000);
                    }
                    
                    // Auto-download sensitivity analysis files if available
                    if (result.sensitivity_csv_url) {
                        setTimeout(() => {
                            const csvLink = document.createElement('a');
                            csvLink.href = result.sensitivity_csv_url;
                            csvLink.download = 'sensitivity_contact_area_results.csv';
                            csvLink.click();
                        }, 2000);
                    }
                    if (result.sensitivity_png_url) {
                        setTimeout(() => {
                            const pngLink = document.createElement('a');
                            pngLink.href = result.sensitivity_png_url;
                            pngLink.download = 'sensitivity_contact_area.png';
                            pngLink.click();
                        }, 3000);
                    }
                } else {
                    logDiv.innerHTML = `<span class="log-error">Error: ${result.error}</span>`;
                    statusBar.className = 'status-bar status-error';
                    statusBar.innerHTML = '❌ Pipeline failed';
                    statusBar.style.display = 'block';
                }
            } catch (error) {
                logDiv.innerHTML = `<span class="log-error">Network error: ${error.message}</span>`;
                statusBar.className = 'status-bar status-error';
                statusBar.innerHTML = '❌ Connection error';
                statusBar.style.display = 'block';
            } finally {
                runBtn.disabled = false;
                runBtn.innerHTML = '🚀 Run Pipeline';
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/run', methods=['POST'])
def run_pipeline():
    try:
        # Get form data
        input_file = request.files.get('input_csv')
        if not input_file:
            return jsonify({'error': 'No input file provided'}), 400
        
        report_out = request.form.get('report_out', 'fit_report.txt')
        phase_code_str = request.form.get('phase_code', '6')
        fill_volume_ul = float(request.form.get('fill_volume', str(ANCHOR_FILL_VOLUME_UL)))
        calib_str = request.form.get('calibration', '')

        # Anchored geometry (E-GEOM-ANCHOR-001) is the only mode for this batch;
        # the locked checkbox exists for display, and the profile select is the
        # GUI equivalent of CLI --geometry-profile.
        anchored_mode = True
        geometry_profile = request.form.get('geometry_profile', 'cone_tip')
        if geometry_profile not in GEOMETRY_PROFILES:
            return jsonify({
                'error': f"Invalid geometry profile {geometry_profile!r}; "
                         f"must be one of {list(GEOMETRY_PROFILES)}"
            }), 400

        try:
            contact_efficiency = float(request.form.get('contact_efficiency', '0.22'))
        except ValueError:
            return jsonify({'error': 'Invalid contact_efficiency value'}), 400
        if not (0.0 < contact_efficiency <= 1.0):
            return jsonify({'error': 'contact_efficiency must be in (0, 1]'}), 400

        use_min_temp = request.form.get('use_min_temp') == 'on'
        use_transient = request.form.get('use_transient') == 'on'
        use_hybrid = request.form.get('hybrid_mode') == 'on'
        plot_residuals = request.form.get('plot_residuals') == 'on'
        run_sensitivity = request.form.get('run_sensitivity') == 'on'
        
        # Parse endpoint detection parameters
        try:
            rolling_window_minutes = float(request.form.get('rolling_window_minutes', '30.0'))
            sustain_minutes = float(request.form.get('sustain_minutes', '30.0'))
        except ValueError as e:
            return jsonify({'error': f'Invalid endpoint detection parameter: {e}'}), 400
        
        log_entries = []
        
        def log(text, tag='info'):
            log_entries.append({'text': text + '\n', 'tag': tag})
        
        log("=" * 60, 'info')
        log("Starting Lyophilization Pipeline", 'info')
        log("=" * 60, 'info')
        
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        input_path = Path(temp_dir) / input_file.filename
        input_file.save(input_path)
        
        log(f"\nInput file: {input_file.filename}", 'info')
        log(f"Primary phase code: {phase_code_str}", 'info')
        log(f"Fill volume: {fill_volume_ul} µL", 'info')

        # --- Geometry mode banner (anchored is the only trusted mode) ------
        log("", 'info')
        log("=" * 68, 'warning')
        log("GEOMETRY MODE: ANCHORED (event E-GEOM-ANCHOR-001)", 'warning')
        log(f"  Measured anchor: {ANCHOR_FILL_VOLUME_UL} uL = {ANCHOR_FILL_HEIGHT_MM} mm fill height", 'warning')
        log(f"  Fill height FIXED at {ANCHOR_FILL_HEIGHT_MM} mm for this batch (no interpolation).", 'warning')
        log(f"  Active area profile (--geometry-profile): {geometry_profile}", 'warning')
        log("  Multi-point calibration DEFERRED (A-GEOM-004) — MANDATORY before", 'warning')
        log("  any recipe optimization or advisory MPC use.", 'warning')
        log("=" * 68, 'warning')

        if calib_str.strip():
            log("", 'error')
            log("*** LOUD WARNING: Calibration points supplied ("
                + calib_str.strip() + ") but anchored mode IGNORES them. ***", 'error')
            log("*** In anchored mode the fill height is FIXED at "
                f"{ANCHOR_FILL_HEIGHT_MM} mm; --calib interpolation is deferred "
                "(A-GEOM-004) and NOT applied. ***", 'error')
            log("", 'error')
        
        log(f"\nOptions:", 'info')
        log(f"  Geometry mode: anchored (fill height fixed at {ANCHOR_FILL_HEIGHT_MM} mm)", 'info')
        log(f"  Geometry profile (--geometry-profile): {geometry_profile}", 'info')
        log(f"  Contact efficiency: {contact_efficiency}", 'info')
        log(f"  Use minimum temp: {use_min_temp}", 'info')
        log(f"  Transient mode: {use_transient}", 'info')
        log(f"  Hybrid mode: {use_hybrid}", 'info')
        log(f"  Plot residuals: {plot_residuals}", 'info')
        log(f"  Run sensitivity analysis: {run_sensitivity}", 'info')
        log(f"  Rolling window: {rolling_window_minutes} min", 'info')
        log(f"  Sustain duration: {sustain_minutes} min", 'info')
        
        # Load and clean data
        log("\n[1/6] Loading and cleaning data...", 'info')
        df = load_raw(input_path)
        df = build_timestamp(df)
        df = normalize_units(df)
        df = flag_primary_drying_end(df)
        log(f"  Loaded {len(df)} rows", 'success')
        
        # Choose model input columns
        log("\n[2/6] Selecting pressure and temperature columns...", 'info')
        pressure_col = choose_pressure_column(df)
        product_temp_col = choose_product_temperature_column(df, use_min=use_min_temp)
        log(f"  Pressure column: {pressure_col}", 'info')
        log(f"  Product temp column: {product_temp_col}", 'info')
        
        # Convert user inputs
        phase_code = coerce_phase_code(df, phase_code_str)
        fill_volume_m3 = fill_volume_ul * 1e-9

        # Anchored geometry (E-GEOM-ANCHOR-001): single measured anchor,
        # selectable area profile. Calibration points are NEVER interpolated
        # in anchored mode — any supplied --calib string is ignored (loud
        # warning already logged above).
        calibration_points = []
        tube = AnchoredTubeGeometry(
            fill_volume_m3=fill_volume_m3,
            fill_height_m=ANCHOR_FILL_HEIGHT_MM * 1e-3,  # fixed at the measured 4.0 mm
            profile=geometry_profile,
            contact_efficiency=contact_efficiency,
        )
        log(f"  AnchoredTubeGeometry(profile={geometry_profile!r}, "
            f"H={tube.H * 1e3:.1f} mm, V(H)={tube.V(tube.H) * 1e6:.2f} uL)", 'success')
        
        # Detect primary drying endpoint using Pirani decline transition
        log("\n[2.5/6] Detecting primary drying endpoint (Pirani decline transition)...", 'info')
        result = find_endpoint(
            df, 
            rolling_window_minutes=rolling_window_minutes,
            sustain_minutes=sustain_minutes,
            min_baseline_fraction=0.25
        )
        endpoint_ts = result[0]
        diagnostics = result[1]
        log(f"  Endpoint timestamp: {endpoint_ts}", 'info')
        if diagnostics.get('primary_baseline') is not None:
            log(f"  Primary drying baseline: {diagnostics['primary_baseline']:.3f} Pa", 'info')
            log(f"  Post-drying baseline: {diagnostics['post_baseline']:.3f} Pa", 'info')
            log(f"  Transition threshold: {diagnostics['threshold']:.3f} Pa", 'info')
            log(f"  Transition detected: {diagnostics['detected']}", 'info')
        
        # Calculate endpoint as time from start of primary drying (in seconds).
        # Mass-balance closure requires this value; silently passing None is forbidden.
        primary_df = df[df["Phase"] == phase_code]
        if endpoint_ts is None or len(primary_df) == 0:
            raise RuntimeError(
                "Cannot compute endpoint_time_s: endpoint detection failed or no "
                "primary-phase rows found. Passing None to the joint fit is "
                "forbidden by the mass-balance closure requirement."
            )
        primary_start_timestamp = primary_df["timestamp"].min()
        if pd.isna(primary_start_timestamp):
            raise RuntimeError(
                "Cannot compute endpoint_time_s: no valid primary-drying start "
                "timestamp available."
            )
        endpoint_time_from_start = (endpoint_ts - primary_start_timestamp).total_seconds()
        if endpoint_time_from_start <= 0:
            raise RuntimeError(
                f"Cannot compute endpoint_time_s: detected endpoint ({endpoint_ts}) "
                f"is not after the start of primary drying ({primary_start_timestamp})."
            )
        log(f"  Endpoint time from start of primary drying: {endpoint_time_from_start:.1f}s ({endpoint_time_from_start/60:.1f} min)", 'info')
        
        # Build segments using centralized physics with variance-based steady-state detection
        log("\n[3/6] Building drying segments...", 'info')
        segments = build_segments(
            df=df,
            primary_phase_code=phase_code,
            tube=tube,
            fill_volume_m3=fill_volume_m3,
            endpoint_timestamp=endpoint_ts,
            shelf_temp_std_threshold=0.1,
            pressure_std_threshold=5.0,
        )
        log(f"  Built {len(segments)} segment(s): {[s.label for s in segments]}", 'success')
        
        # Report detected step types
        if use_hybrid:
            hold_steps = [s.label for s in segments if s.is_hold_step]
            ramp_steps = [s.label for s in segments if not s.is_hold_step]
            log(f"\n[info] Hybrid mode detected:", 'info')
            log(f"  Hold steps (steady-state): {len(hold_steps)} - {hold_steps}", 'info')
            log(f"  Ramp steps (transient): {len(ramp_steps)} - {ramp_steps}", 'info')
        
        if len(segments) < 2:
            log("\n[warning] Only one segment found -- Kv/Rp may not be uniquely identifiable.", 'warning')
        
        # Run joint fit
        log("\n[4/6] Running joint parameter fit...", 'info')
        use_transient_final = use_transient or use_hybrid
        
        fitted = fit_parameters_joint(
            segments,
            use_transient=use_transient_final,
            use_hybrid=use_hybrid,
            endpoint_time_s=endpoint_time_from_start,
            mass_balance_weight=None
        )
        
        if not fitted['converged']:
            log("[warning] Optimization did not converge!", 'warning')
        else:
            log("  Fit converged successfully", 'success')

        # Log the mass-balance closure lines to the web GUI log as well
        for mb_line in build_mass_balance_lines(fitted, endpoint_time_from_start):
            log(f"  {mb_line}", 'info')
        
        log(f"\nFitted parameters:", 'info')
        log(f"  Kv  = {fitted['Kv']:.6f}", 'info')
        log(f"  Rp0 = {fitted['Rp0']:.6f}", 'info')
        log(f"  A1  = {fitted['A1']:.6f}", 'info')
        log(f"  A2  = {fitted['A2']:.6f}", 'info')
        log(f"  Overall RMS error: {fitted['rms_error_k']:.6f} K", 'success')
        
        # Build report
        log("\n[5/6] Generating report...", 'info')
        lines = [
            "=== Joint Pikal model fit ===",
            f"Input file: {input_file.filename}",
            f"Primary phase code used: {phase_code}",
            f"Fill volume: {fill_volume_ul} uL",
            "",
            "=== Geometry (E-GEOM-ANCHOR-001: single measured anchor) ===",
            "Geometry mode: ANCHORED — fill height FIXED at 4.0 mm (no --calib interpolation)",
            "Measured anchor: 16.0 uL = 4.0 mm fill height (production PCR tubes)",
            f"Active area profile (--geometry-profile): {geometry_profile}",
            "Multi-point calibration: DEFERRED (A-GEOM-004) — MANDATORY before any",
            "recipe optimization or advisory MPC use.",
            "Ice mass is DERIVED from L only: ice_mass = RHO_ICE * (V(H) - V(H - L)).",
            f"Simulated endpoint definition: first time L >= H ({ANCHOR_FILL_HEIGHT_MM:.1f} mm); no extrapolation fallback.",
            "",
            f"Geometry table for active profile '{geometry_profile}' (h vs V(h) vs A(h), 0.5 mm steps):",
            f"  {'h (mm)':>8} | {'V (uL)':>12} | {'A (mm^2)':>12}",
            "  " + "-" * 40,
        ]
        for h_m, v_m3, a_m2 in build_geometry_table_rows(tube, h_step_m=0.5e-3):
            lines.append(
                f"  {h_m * 1e3:>8.1f} | {v_m3 * 1e6:>12.4f} | {a_m2 * 1e6:>12.4f}"
            )
        lines += [
            "",
            f"Pressure column used: {pressure_col}",
            f"Product temperature column used: {product_temp_col}",
            f"Using minimum probe temperature (ice-front approx): {use_min_temp}",
            f"Using transient mode (for multi-step ramps): {use_transient and not use_hybrid}",
            f"Using hybrid mode (shelf-setpt based Hold/Ramp detection): {use_hybrid}",
            f"Contact efficiency: {contact_efficiency}",
            f"Calibration points used: {len(calibration_points)} (anchored mode ignores --calib)",
            "",
            "Detected recipe setpoints (Hold vs Ramp):",
        ]
        
        for seg in segments:
            step_type = "Hold (steady-state)" if seg.is_hold_step else "Ramp (transient)"
            Ts_min = seg.Ts_k.min() - 273.15
            Ts_max = seg.Ts_k.max() - 273.15
            Pc_mean = seg.Pc_pa.mean() / 100.0
            
            lines.append(f"  {seg.label}: {step_type}")
            lines.append(f"    Shelf temp: {Ts_min:.1f}C to {Ts_max:.1f}C")
            lines.append(f"    Chamber pressure: {Pc_mean:.2f} mbar")
            lines.append(f"    Duration: {seg.t_s[-1]:.0f}s ({seg.t_s[-1]/60:.1f} min)")
        
        lines.extend([
            "",
            "=== Endpoint Detection (Pirani Decline Transition) ===",
            f"Primary drying baseline (median Pirani-CM diff, early phase): {diagnostics.get('primary_baseline', 'N/A'):.3f} Pa" if diagnostics.get('primary_baseline') is not None else "",
            f"Post-drying baseline (median Pirani-CM diff, late phase): {diagnostics.get('post_baseline', 'N/A'):.3f} Pa" if diagnostics.get('post_baseline') is not None else "",
            f"Transition threshold used: {diagnostics.get('threshold', 'N/A'):.3f} Pa" if diagnostics.get('threshold') is not None else "",
            f"Detected endpoint timestamp: {endpoint_ts}",
            f"Endpoint time from start of primary drying: {endpoint_time_from_start:.1f} s" if endpoint_time_from_start is not None else "",
            f"Transition detected: {diagnostics.get('detected', False)}",
            "",
            "=== Mass-Balance Constraint ===",
        ] + build_mass_balance_lines(fitted, endpoint_time_from_start) + [
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

        # Per-segment residual-sign / probe-spread diagnostics (task 6):
        # mean & median residual sign (simulated minus measured) and the
        # median probe spread (max-min of TP01..TP04).
        seg_diags = fitted.get("per_segment_diagnostics", {})
        if seg_diags:
            lines += [
                "",
                "Per-segment residual diagnostics (residual = simulated - measured):",
                f"  {'segment':<28} {'mean (K)':>10} {'median (K)':>11} {'sign':>9} {'med spread (K)':>15}",
                "  " + "-" * 78,
            ]
            for label, d in seg_diags.items():
                spread = d.get("median_probe_spread_k")
                spread_str = f"{spread:.3f}" if spread is not None else "n/a"
                lines.append(
                    f"  {label:<28} {d['mean_residual_k']:>+10.4f} "
                    f"{d['median_residual_k']:>+11.4f} "
                    f"{d['residual_sign']:>9} {spread_str:>15}"
                )
        
        # Add state continuity diagnostics if using continuous simulation
        if use_transient_final or use_hybrid:
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
                    segments, fitted['Kv'], fitted['Rp0'], fitted['A1'], fitted['A2'],
                    use_hybrid=use_hybrid
                )
            state_output = f.getvalue()
            lines.append(state_output)
        
        report = "\n".join(lines)
        report_path = Path(temp_dir) / report_out
        report_path.write_text(report, encoding="utf-8")
        log(f"  Saved report -> {report_out}", 'success')
        
        # Generate residual plots if requested
        if plot_residuals:
            log("\n[6/6] Generating residual plots...", 'info')
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                if use_transient_final or use_hybrid:
                    all_Tp_sim = simulate_continuous_primary_drying(
                        segments, fitted['Kv'], fitted['Rp0'], fitted['A1'], fitted['A2'],
                        use_hybrid=use_hybrid
                    )
                else:
                    all_Tp_sim = []
                    for seg in segments:
                        Tp_sim = simulate_cycle(
                            seg.t_s, seg.Ts_k, seg.Pc_pa, seg.tube,
                            seg.fill_volume_m3, fitted['Kv'], fitted['Rp0'],
                            fitted['A1'], fitted['A2'], use_transient=False
                        )
                        all_Tp_sim.append(Tp_sim)
                
                for i, seg in enumerate(segments):
                    Tp_sim = all_Tp_sim[i]
                    residuals = Tp_sim - seg.Tp_measured_k
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                    
                    ax1.plot(seg.t_s / 60, seg.Tp_measured_k - 273.15, 'b-', 
                            label='Measured', linewidth=1.5)
                    ax1.plot(seg.t_s / 60, Tp_sim - 273.15, 'r--', 
                            label='Simulated', linewidth=1.5)
                    ax1.set_ylabel('Temperature (°C)')
                    step_type = "Hold (steady-state)" if seg.is_hold_step else "Ramp (transient)"
                    continuity_note = " [continuous]" if (use_transient_final or use_hybrid) else ""
                    ax1.set_title(f'{seg.label} - Step Type: {step_type}{continuity_note}')
                    ax1.legend(loc='best')
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(seg.t_s / 60, residuals, 'g-', linewidth=1)
                    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                    ax2.fill_between(seg.t_s / 60, residuals, 0, alpha=0.3, color='green')
                    ax2.set_xlabel('Time (min)')
                    ax2.set_ylabel('Residual (°C)')
                    ax2.set_title(f'Residuals (Simulated - Measured), RMS = {np.sqrt(np.mean(residuals**2)):.3f} K')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    plot_filename = f"residual_{seg.label.replace('=', '_').replace('cycle_', 'c').replace('step_', 's')}.png"
                    plot_path = Path(temp_dir) / plot_filename
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    log(f"  Saved: {plot_filename}", 'success')
                    plt.close(fig)
                
                log("Residual plots generated successfully.", 'success')
            except ImportError:
                log("[warning] matplotlib not installed; cannot generate residual plots.", 'warning')
            except Exception as exc:
                log(f"[warning] Failed to generate residual plots: {exc}", 'warning')
        
        sensitivity_csv_filename = None
        sensitivity_png_filename = None

        # Run contact area sensitivity analysis if requested
        if run_sensitivity:
            log("\n[7/7] Running contact area sensitivity analysis...", 'info')
            try:
                from sensitivity_contact_area import (
                    print_summary_table,
                    find_optimal_contact_efficiency,
                    generate_plot,
                )

                # Anchored-mode sweep: rebuilds an AnchoredTubeGeometry with the
                # ACTIVE profile for every contact efficiency value (the legacy
                # run_sensitivity_analysis() still uses CalibratedPCRTubeGeometry,
                # which contradicts E-GEOM-ANCHOR-001).
                sens_results_df = _anchored_sensitivity_fit(
                    input_path=input_path,
                    primary_phase_code=phase_code,
                    fill_volume_ul=fill_volume_ul,
                    contact_efficiencies=CONTACT_EFFICIENCY_VALUES,
                    geometry_profile=geometry_profile,
                    use_min_temp=use_min_temp,
                    use_transient=use_transient_final,
                    use_hybrid=use_hybrid,
                )
                sens_endpoint_diagnostics = {}
                log(f"  Sweep mode: ANCHORED profile={geometry_profile} "
                    f"(contact efficiency values: {CONTACT_EFFICIENCY_VALUES})", 'info')
                
                # Print summary table to log
                log("\n=== CONTACT EFFICIENCY SENSITIVITY ANALYSIS ===", 'info')
                valid_df = sens_results_df[sens_results_df['error'].isna()].copy()
                if not valid_df.empty:
                    header = (
                        f"{'Contact Eff.':>14} | {'Kv':>12} | {'Rp0':>10} | {'A1':>10} | {'A2':>10} | "
                        f"{'Overall RMS':>12} | {'Step 4 RMS':>12} | {'Total HTC':>12}"
                    )
                    log(header, 'info')
                    log("-" * 100, 'info')
                    
                    for _, row in sens_results_df.iterrows():
                        if row['error']:
                            log(f"{row['contact_efficiency']:>14.2f} | FIT ERROR: {row['error'][:50]}", 'warning')
                        else:
                            line = (
                                f"{row['contact_efficiency']:>14.2f} | {row['Kv']:>12.6f} | {row['Rp0']:>10.6f} | "
                                f"{row['A1']:>10.6f} | {row['A2']:>10.6f} | {row['overall_rms_k']:>12.4f} | "
                                f"{row['step_4_rms_k']:>12.4f} | {row['total_htc']:>12.6f}"
                            )
                            log(line, 'info')
                    
                    log("=" * 100, 'info')
                    
                    # Find optimal
                    optimal_row = find_optimal_contact_efficiency(sens_results_df)
                    if optimal_row is not None:
                        optimal_value = optimal_row['contact_efficiency']
                        metric_col = 'step_4_rms_k' if not np.isnan(optimal_row['step_4_rms_k']) else 'overall_rms_k'
                        metric_name = 'Step 4 RMS' if not np.isnan(optimal_row['step_4_rms_k']) else 'Overall RMS'
                        log(f"\n*** OPTIMAL CONTACT EFFICIENCY: {optimal_value:.2f} ***", 'success')
                        log(f"    Minimizes {metric_name}: {optimal_row[metric_col]:.4f} K", 'info')
                        log(f"    Corresponding Kv: {optimal_row['Kv']:.6f} W/m²/K", 'info')
                        
                        # Generate plot
                        plot_out = Path(temp_dir) / "sensitivity_contact_area.png"
                        csv_out = Path(temp_dir) / "sensitivity_contact_area_results.csv"
                        
                        sens_results_df.to_csv(csv_out, index=False)
                        log(f"\nSaved results -> sensitivity_contact_area_results.csv", 'success')
                        
                        generate_plot(sens_results_df, plot_out, optimal_value)
                        log(f"Saved plot -> sensitivity_contact_area.png", 'success')
                        
                        # Store filenames for download
                        sensitivity_csv_filename = "sensitivity_contact_area_results.csv"
                        sensitivity_png_filename = "sensitivity_contact_area.png"
                else:
                    log("[warning] No valid results from sensitivity analysis.", 'warning')
                    
            except ImportError as e:
                log(f"[warning] Cannot import sensitivity_contact_area: {e}", 'warning')
            except Exception as exc:
                log(f"[warning] Sensitivity analysis failed: {exc}", 'warning')
        
        log("\n" + "=" * 60, 'success')
        log("Pipeline completed successfully!", 'success')
        log("=" * 60, 'success')
        
        # Clean up temp file
        try:
            input_path.unlink()
        except:
            pass
        
        # Build response with optional sensitivity file URLs
        response_data = {
            'log': log_entries,
            'report_url': f'/download/{report_out}'
        }
        if sensitivity_csv_filename and sensitivity_png_filename:
            response_data['sensitivity_csv_url'] = f'/download/{sensitivity_csv_filename}'
            response_data['sensitivity_png_url'] = f'/download/{sensitivity_png_filename}'
        
        return jsonify(response_data)
        
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/download/<filename>')
def download(filename):
    filepath = Path(tempfile.gettempdir()) / filename
    if filepath.exists():
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 Starting Lyophilization Pipeline Web GUI")
    print("=" * 60)
    print("\nOpen your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
