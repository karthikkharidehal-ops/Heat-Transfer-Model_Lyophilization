"""
GUI for Lyophilization Pikal Model Pipeline

Provides a user-friendly interface for running the Pikal model fitting pipeline
with all options available as toggles and input fields.
"""

import sys
import threading
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Entry, Button, Checkbutton, IntVar, StringVar, 
    Text, Scrollbar, END, filedialog, messagebox, ttk
)
from tkinter.scrolledtext import ScrolledText

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


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lyophilization Pikal Model Pipeline")
        self.root.geometry("900x700")
        
        # Variables
        self.input_csv = StringVar()
        self.phase_code = StringVar(value="6")
        self.fill_volume = StringVar(value="12")
        self.calibration = StringVar()
        self.report_out = StringVar(value="fit_report.txt")
        
        self.use_min_temp = IntVar(value=0)
        self.use_transient = IntVar(value=0)
        self.hybrid_mode = IntVar(value=0)
        self.plot_residuals = IntVar(value=0)
        
        # Endpoint detection parameters
        self.rolling_window_minutes = StringVar(value="30.0")
        self.sustain_minutes = StringVar(value="30.0")
        
        self.running = False
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the user interface."""
        # Main container with scrollbar
        main_frame = Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        canvas = None  # We'll use a simple frame approach instead
        
        # === File Selection Section ===
        file_frame = LabelFrame(main_frame, text="Input/Output Files", padx=10, pady=10)
        file_frame.pack(fill='x', pady=(0, 10))
        
        # Input CSV
        Label(file_frame, text="Input CSV:").grid(row=0, column=0, sticky='w', pady=5)
        Entry(file_frame, textvariable=self.input_csv, width=50).grid(row=0, column=1, padx=5, pady=5)
        Button(file_frame, text="Browse...", command=self._browse_input).grid(row=0, column=2, pady=5)
        
        # Report output
        Label(file_frame, text="Report Output:").grid(row=1, column=0, sticky='w', pady=5)
        Entry(file_frame, textvariable=self.report_out, width=50).grid(row=1, column=1, padx=5, pady=5)
        Button(file_frame, text="Browse...", command=self._browse_output).grid(row=1, column=2, pady=5)
        
        # === Model Parameters Section ===
        param_frame = LabelFrame(main_frame, text="Model Parameters", padx=10, pady=10)
        param_frame.pack(fill='x', pady=(0, 10))
        
        # Phase code
        Label(param_frame, text="Primary Phase Code:").grid(row=0, column=0, sticky='w', pady=5)
        Entry(param_frame, textvariable=self.phase_code, width=20).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        Label(param_frame, text="(e.g., 6 for primary drying)").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        
        # Fill volume
        Label(param_frame, text="Fill Volume (µL):").grid(row=1, column=0, sticky='w', pady=5)
        Entry(param_frame, textvariable=self.fill_volume, width=20).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        Label(param_frame, text="(e.g., 6, 12, 20)").grid(row=1, column=2, sticky='w', padx=5, pady=5)
        
        # Calibration points
        Label(param_frame, text="Calibration Points:").grid(row=2, column=0, sticky='nw', pady=5)
        calib_entry = Entry(param_frame, textvariable=self.calibration, width=50)
        calib_entry.grid(row=2, column=1, columnspan=2, sticky='w', padx=5, pady=5)
        Label(param_frame, text="Format: vol_ul:height_mm (e.g., 6:1.8 12:2.9 20:4.1)", 
              fg='gray').grid(row=3, column=1, columnspan=2, sticky='w', padx=5)
        
        # === Options Section ===
        options_frame = LabelFrame(main_frame, text="Simulation Options", padx=10, pady=10)
        options_frame.pack(fill='x', pady=(0, 10))
        
        # Create checkboxes in a grid
        Checkbutton(options_frame, text="Use minimum probe temperature (ice-front approximation)",
                   variable=self.use_min_temp).grid(row=0, column=0, sticky='w', pady=5, padx=5)
        
        Checkbutton(options_frame, text="Transient mode (for multi-step ramps)",
                   variable=self.use_transient).grid(row=1, column=0, sticky='w', pady=5, padx=5)
        
        Checkbutton(options_frame, text="Hybrid mode (auto-detect Hold vs Ramp steps)",
                   variable=self.hybrid_mode).grid(row=2, column=0, sticky='w', pady=5, padx=5)
        
        Checkbutton(options_frame, text="Generate residual plots (PNG files)",
                   variable=self.plot_residuals).grid(row=3, column=0, sticky='w', pady=5, padx=5)
        
        # Endpoint detection parameters
        Label(options_frame, text="Endpoint Detection Parameters:", font=('Arial', 10, 'bold')
              ).grid(row=4, column=0, sticky='w', pady=(10, 5), padx=5)
        
        Label(options_frame, text="Rolling Window (min):").grid(row=5, column=0, sticky='w', pady=5, padx=5)
        Entry(options_frame, textvariable=self.rolling_window_minutes, width=10).grid(row=5, column=1, sticky='w', padx=5, pady=5)
        
        Label(options_frame, text="Sustain Duration (min):").grid(row=6, column=0, sticky='w', pady=5, padx=5)
        Entry(options_frame, textvariable=self.sustain_minutes, width=10).grid(row=6, column=1, sticky='w', padx=5, pady=5)
        
        # Help text for hybrid mode
        help_text = ("Hybrid mode automatically detects Hold steps (steady shelf temp) vs "
                    "Ramp steps (changing shelf temp) and applies appropriate physics.")
        Label(options_frame, text=help_text, fg='blue', wraplength=600, justify='left'
              ).grid(row=7, column=0, sticky='w', padx=5, pady=(0, 5))
        
        # === Run Button Section ===
        button_frame = Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))
        
        self.run_button = Button(button_frame, text="Run Pipeline", command=self._run_pipeline,
                                bg='green', fg='white', font=('Arial', 12, 'bold'), padx=20, pady=5)
        self.run_button.pack(side='left', padx=5)
        
        Button(button_frame, text="Clear Log", command=self._clear_log,
              padx=10, pady=5).pack(side='right', padx=5)
        
        # === Output Log Section ===
        log_frame = LabelFrame(main_frame, text="Execution Log", padx=10, pady=10)
        log_frame.pack(fill='both', expand=True)
        
        self.log_text = ScrolledText(log_frame, wrap='word', height=20, font=('Courier', 9))
        self.log_text.pack(fill='both', expand=True)
        
        # Configure text tags for coloring
        self.log_text.tag_configure('error', foreground='red')
        self.log_text.tag_configure('warning', foreground='orange')
        self.log_text.tag_configure('success', foreground='green')
        self.log_text.tag_configure('info', foreground='blue')
    
    def _browse_input(self):
        """Open file dialog to select input CSV."""
        filename = filedialog.askopenfilename(
            title="Select Input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_csv.set(filename)
    
    def _browse_output(self):
        """Open file dialog to select report output location."""
        filename = filedialog.asksaveasfilename(
            title="Save Report As",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.report_out.set(filename)
    
    def _log(self, message, tag=None):
        """Add message to log."""
        self.log_text.insert(END, message + '\n', tag)
        self.log_text.see(END)
        self.root.update_idletasks()
    
    def _clear_log(self):
        """Clear the log window."""
        self.log_text.delete(1.0, END)
    
    def _run_pipeline(self):
        """Run the pipeline in a separate thread."""
        if self.running:
            messagebox.showwarning("Pipeline Running", 
                                  "A pipeline execution is already in progress.")
            return
        
        # Validate inputs
        if not self.input_csv.get():
            messagebox.showerror("Error", "Please select an input CSV file.")
            return
        
        if not Path(self.input_csv.get()).exists():
            messagebox.showerror("Error", f"Input file not found: {self.input_csv.get()}")
            return
        
        try:
            fill_vol = float(self.fill_volume.get())
            if fill_vol <= 0:
                raise ValueError("Fill volume must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid fill volume: {e}")
            return
        
        self.running = True
        self.run_button.config(state='disabled', bg='gray')
        self._clear_log()
        
        # Run in background thread
        thread = threading.Thread(target=self._execute_pipeline, daemon=True)
        thread.start()
    
    def _execute_pipeline(self):
        """Execute the pipeline (runs in background thread)."""
        try:
            self._log("=" * 60)
            self._log("Starting Lyophilization Pipeline", 'info')
            self._log("=" * 60)
            
            # Parse inputs
            input_path = Path(self.input_csv.get())
            report_path = Path(self.report_out.get())
            phase_code_str = self.phase_code.get()
            fill_volume_ul = float(self.fill_volume.get())
            calib_str = self.calibration.get()
            
            use_min_temp = bool(self.use_min_temp.get())
            use_transient = bool(self.use_transient.get())
            use_hybrid = bool(self.hybrid_mode.get())
            plot_residuals = bool(self.plot_residuals.get())
            
            # Parse endpoint detection parameters
            try:
                rolling_window_minutes = float(self.rolling_window_minutes.get())
                sustain_minutes = float(self.sustain_minutes.get())
            except ValueError as e:
                self._log(f"[error] Invalid endpoint detection parameter: {e}", 'error')
                self.running = False
                self.run_button.config(state='normal', bg='green')
                return
            
            self._log(f"\nInput file: {input_path.name}")
            self._log(f"Primary phase code: {phase_code_str}")
            self._log(f"Fill volume: {fill_volume_ul} µL")
            
            if calib_str.strip():
                self._log(f"Calibration points: {calib_str}")
            
            self._log(f"\nOptions:")
            self._log(f"  Use minimum temp: {use_min_temp}")
            self._log(f"  Transient mode: {use_transient}")
            self._log(f"  Hybrid mode: {use_hybrid}")
            self._log(f"  Plot residuals: {plot_residuals}")
            self._log(f"  Rolling window: {rolling_window_minutes} min")
            self._log(f"  Sustain duration: {sustain_minutes} min")
            
            # Load and clean data
            self._log("\n[1/6] Loading and cleaning data...")
            df = load_raw(input_path)
            df = build_timestamp(df)
            df = normalize_units(df)
            df = flag_primary_drying_end(df)
            self._log(f"  Loaded {len(df)} rows", 'success')
            
            # Choose model input columns
            self._log("\n[2/6] Selecting pressure and temperature columns...")
            pressure_col = choose_pressure_column(df)
            product_temp_col = choose_product_temperature_column(df, use_min=use_min_temp)
            self._log(f"  Pressure column: {pressure_col}", 'info')
            self._log(f"  Product temp column: {product_temp_col}", 'info')
            
            # Convert user inputs
            phase_code = coerce_phase_code(df, phase_code_str)
            fill_volume_m3 = fill_volume_ul * 1e-9
            
            # Parse calibration
            calibration_points = parse_calib(calib_str) if calib_str.strip() else []
            self._log(f"  Calibration points parsed: {len(calibration_points)}", 'info')
            
            # Use calibrated geometry
            tube = CalibratedPCRTubeGeometry(calibration_points=calibration_points)
            
            # Detect primary drying endpoint using Pirani decline transition
            self._log("\n[2.5/6] Detecting primary drying endpoint (Pirani decline transition)...")
            result = find_endpoint(
                df, 
                rolling_window_minutes=rolling_window_minutes,
                sustain_minutes=sustain_minutes,
                min_baseline_fraction=0.25
            )
            endpoint_ts = result[0]
            diagnostics = result[1]
            self._log(f"  Endpoint timestamp: {endpoint_ts}", 'info')
            if diagnostics.get('primary_baseline') is not None:
                self._log(f"  Primary drying baseline: {diagnostics['primary_baseline']:.3f} Pa", 'info')
                self._log(f"  Post-drying baseline: {diagnostics['post_baseline']:.3f} Pa", 'info')
                self._log(f"  Transition threshold: {diagnostics['threshold']:.3f} Pa", 'info')
                self._log(f"  Transition detected: {diagnostics['detected']}", 'info')
            
            # Calculate endpoint time in seconds from start of primary drying.
            # Mass-balance closure requires this value; silently passing None is forbidden.
            primary_df = df[df["Phase"] == phase_code].copy()
            if endpoint_ts is None or len(primary_df) == 0:
                raise RuntimeError(
                    "Cannot compute endpoint_time_s: endpoint detection failed or no "
                    "primary-phase rows found. Passing None to the joint fit is "
                    "forbidden by the mass-balance closure requirement."
                )
            first_timestamp = primary_df["timestamp"].min()
            endpoint_time_s = (endpoint_ts - first_timestamp).total_seconds()
            if endpoint_time_s <= 0:
                raise RuntimeError(
                    f"Cannot compute endpoint_time_s: detected endpoint ({endpoint_ts}) "
                    f"is not after the start of primary drying ({first_timestamp})."
                )
            self._log(f"  Endpoint time from start of primary drying: {endpoint_time_s:.1f} s", 'info')
            
            # Build segments using centralized physics with variance-based steady-state detection
            self._log("\n[3/6] Building drying segments...")
            segments = build_segments(
                df=df,
                primary_phase_code=phase_code,
                tube=tube,
                fill_volume_m3=fill_volume_m3,
                endpoint_timestamp=endpoint_ts,
                shelf_temp_std_threshold=0.1,
                pressure_std_threshold=5.0,
            )
            self._log(f"  Built {len(segments)} segment(s): {[s.label for s in segments]}", 'success')
            
            # Report detected step types
            if use_hybrid:
                hold_steps = [s.label for s in segments if s.is_hold_step]
                ramp_steps = [s.label for s in segments if not s.is_hold_step]
                self._log(f"\n[info] Hybrid mode detected:", 'info')
                self._log(f"  Hold steps (steady-state): {len(hold_steps)} - {hold_steps}", 'info')
                self._log(f"  Ramp steps (transient): {len(ramp_steps)} - {ramp_steps}", 'info')
            
            if len(segments) < 2:
                self._log("\n[warning] Only one segment found -- Kv/Rp may not be uniquely identifiable.", 'warning')
            
            # Run joint fit
            self._log("\n[4/6] Running joint parameter fit...")
            use_transient_final = use_transient or use_hybrid
            
            fitted = fit_parameters_joint(
                segments,
                use_transient=use_transient_final,
                use_hybrid=use_hybrid,
                endpoint_time_s=endpoint_time_s,
                mass_balance_weight=None
            )
            
            if not fitted['converged']:
                self._log("[warning] Optimization did not converge!", 'warning')
            else:
                self._log("  Fit converged successfully", 'success')
            
            # Log the mass-balance closure lines to the GUI log as well
            for mb_line in build_mass_balance_lines(fitted, endpoint_time_s):
                self._log(f"  {mb_line}")

            self._log("\nFitted parameters:")
            self._log(f"  Kv  = {fitted['Kv']:.6f}")
            self._log(f"  Rp0 = {fitted['Rp0']:.6f}")
            self._log(f"  A1  = {fitted['A1']:.6f}")
            self._log(f"  A2  = {fitted['A2']:.6f}")
            self._log(f"  Overall RMS error: {fitted['rms_error_k']:.6f} K", 'success')
            
            # Build report
            self._log("\n[5/6] Generating report...")
            lines = [
                "=== Joint Pikal model fit ===",
                f"Input file: {input_path.name}",
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
                "=== Endpoint Detection (Pirani Decline Transition) ===",
                f"Primary drying baseline (median Pirani-CM diff, early phase): {diagnostics.get('primary_baseline', 'N/A'):.3f} Pa" if diagnostics.get('primary_baseline') is not None else "",
                f"Post-drying baseline (median Pirani-CM diff, late phase): {diagnostics.get('post_baseline', 'N/A'):.3f} Pa" if diagnostics.get('post_baseline') is not None else "",
                f"Transition threshold used: {diagnostics.get('threshold', 'N/A'):.3f} Pa" if diagnostics.get('threshold') is not None else "",
                f"Detected endpoint timestamp: {endpoint_ts}",
                f"Endpoint time from start of primary drying: {endpoint_time_s:.1f} s" if endpoint_time_s is not None else "",
                f"Transition detected: {diagnostics.get('detected', False)}",
                "",
                "=== Mass-Balance Constraint ===",
            ] + build_mass_balance_lines(fitted, endpoint_time_s) + [
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
            report_path.write_text(report, encoding="utf-8")
            self._log(f"  Saved report -> {report_path}", 'success')
            
            # Generate residual plots if requested
            if plot_residuals:
                self._log("\n[6/6] Generating residual plots...")
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    # Use continuous simulation across all segments when appropriate
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
                        
                        # Top panel: Measured vs Simulated
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
                        
                        # Bottom panel: Residuals
                        ax2.plot(seg.t_s / 60, residuals, 'g-', linewidth=1)
                        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                        ax2.fill_between(seg.t_s / 60, residuals, 0, alpha=0.3, color='green')
                        ax2.set_xlabel('Time (min)')
                        ax2.set_ylabel('Residual (°C)')
                        ax2.set_title(f'Residuals (Simulated - Measured), RMS = {np.sqrt(np.mean(residuals**2)):.3f} K')
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        plot_filename = f"residual_{seg.label.replace('=', '_').replace('cycle_', 'c').replace('step_', 's')}.png"
                        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        self._log(f"  Saved: {plot_filename}", 'success')
                        plt.close(fig)
                    
                    self._log("Residual plots generated successfully.", 'success')
                except ImportError:
                    self._log("[warning] matplotlib not installed; cannot generate residual plots.", 'warning')
                    self._log("Install with: pip install matplotlib", 'warning')
                except Exception as exc:
                    self._log(f"[warning] Failed to generate residual plots: {exc}", 'warning')
            
            self._log("\n" + "=" * 60)
            self._log("Pipeline completed successfully!", 'success')
            self._log("=" * 60)
            
            messagebox.showinfo("Success", "Pipeline completed successfully!\n\nCheck the log for details and the output files.")
            
        except Exception as exc:
            self._log(f"\n[error] Pipeline failed: {exc}", 'error')
            messagebox.showerror("Error", f"Pipeline failed:\n{exc}")
        
        finally:
            self.running = False
            self.run_button.config(state='normal', bg='green')


def main():
    root = Tk()
    app = PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
