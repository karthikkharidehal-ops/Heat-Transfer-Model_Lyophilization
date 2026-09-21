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
)

from pikal_model import fit_parameters_joint, simulate_cycle, simulate_continuous_primary_drying


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
        input[type="text"], input[type="number"] {
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
                <h2>⚙️ Model Parameters</h2>
                <div class="form-group">
                    <label for="phase_code">Primary Phase Code:</label>
                    <input type="text" id="phase_code" name="phase_code" value="6" placeholder="e.g., 6">
                    <div class="help-text">The value in the Phase column that indicates primary drying</div>
                </div>
                <div class="form-group">
                    <label for="fill_volume">Fill Volume (µL):</label>
                    <input type="number" id="fill_volume" name="fill_volume" value="12" step="0.1" min="0">
                    <div class="help-text">Example values: 6, 12, 20 µL</div>
                </div>
                <div class="form-group">
                    <label for="calibration">Calibration Points (optional):</label>
                    <input type="text" id="calibration" name="calibration" placeholder="6:1.8 12:2.9 20:4.1">
                    <div class="help-text">Format: vol_ul:height_mm (space-separated)</div>
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
        fill_volume_ul = float(request.form.get('fill_volume', '12'))
        calib_str = request.form.get('calibration', '')
        
        use_min_temp = request.form.get('use_min_temp') == 'on'
        use_transient = request.form.get('use_transient') == 'on'
        use_hybrid = request.form.get('hybrid_mode') == 'on'
        plot_residuals = request.form.get('plot_residuals') == 'on'
        
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
        
        if calib_str.strip():
            log(f"Calibration points: {calib_str}", 'info')
        
        log(f"\nOptions:", 'info')
        log(f"  Use minimum temp: {use_min_temp}", 'info')
        log(f"  Transient mode: {use_transient}", 'info')
        log(f"  Hybrid mode: {use_hybrid}", 'info')
        log(f"  Plot residuals: {plot_residuals}", 'info')
        
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
        
        # Parse calibration
        calibration_points = parse_calib(calib_str) if calib_str.strip() else []
        log(f"  Calibration points parsed: {len(calibration_points)}", 'info')
        
        # Use calibrated geometry
        tube = CalibratedPCRTubeGeometry(calibration_points=calibration_points)
        
        # Build segments using centralized physics with variance-based steady-state detection
        log("\n[3/6] Building drying segments...", 'info')
        segments = build_segments(
            df=df,
            primary_phase_code=phase_code,
            tube=tube,
            fill_volume_m3=fill_volume_m3,
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
            use_hybrid=use_hybrid
        )
        
        if not fitted['converged']:
            log("[warning] Optimization did not converge!", 'warning')
        else:
            log("  Fit converged successfully", 'success')
        
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
            f"Pressure column used: {pressure_col}",
            f"Product temperature column used: {product_temp_col}",
            f"Using minimum probe temperature (ice-front approx): {use_min_temp}",
            f"Using transient mode (for multi-step ramps): {use_transient and not use_hybrid}",
            f"Using hybrid mode (shelf-setpt based Hold/Ramp detection): {use_hybrid}",
            f"Calibration points used: {len(calibration_points)}",
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
        
        log("\n" + "=" * 60, 'success')
        log("Pipeline completed successfully!", 'success')
        log("=" * 60, 'success')
        
        # Clean up temp file
        try:
            input_path.unlink()
        except:
            pass
        
        return jsonify({
            'log': log_entries,
            'report_url': f'/download/{report_out}'
        })
        
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
