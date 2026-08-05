# GUI Options for Lyophilization Pipeline

I've created **two GUI options** to replace the command-line interface:

## Option 1: Web-Based GUI (Recommended) ✅

Since tkinter is not available in this environment, I've created a **web-based GUI** that runs in your browser.

### To Start the Web GUI:

```bash
python gui_web.py
```

Then open your browser to: **http://localhost:5000**

### Features:
- 📁 File upload via drag-and-drop or browse button
- ⚙️ All model parameters with input validation
- 🔬 Toggle switches for all simulation options:
  - Use minimum probe temperature
  - Transient mode
  - **Hybrid mode (auto-detect Hold vs Ramp)** ⭐ Recommended by default
  - Generate residual plots
- 📊 Real-time execution log with color-coded messages
- 💾 Automatic report download after completion
- 🎨 Modern, responsive design

---

## Option 2: Desktop GUI (Requires Tkinter)

If you have tkinter installed on your local machine, you can use the desktop version:

```bash
python gui_pipeline.py
```

This provides the same functionality as the web version but as a standalone desktop application.

---

## Comparison: CLI vs GUI

| Feature | CLI Command | GUI Equivalent |
|---------|-------------|----------------|
| Input file | `export.csv` | File upload/browse |
| Phase code | `--primary-phase-code 6` | Text input field |
| Fill volume | `--fill-volume-ul 12` | Number input field |
| Calibration | `--calib 6:1.8 12:2.9` | Text input field |
| Min temp | `--use-min-temp` | Checkbox |
| Transient | `--use-transient` | Checkbox |
| Hybrid mode | `--hybrid-mode` | Checkbox (checked by default) |
| Residual plots | `--plot-residuals` | Checkbox |
| Report output | `--report-out fit_report.txt` | Text input field |

---

## Example Migration

### Old CLI Command:
```bash
python run_pipeline.py export.csv \
  --primary-phase-code 6 \
  --fill-volume-ul 12 \
  --calib 6:1.8 12:2.9 20:4.1 \
  --hybrid-mode \
  --plot-residuals
```

### New GUI Workflow:
1. Run `python gui_web.py`
2. Open http://localhost:5000
3. Upload `export.csv`
4. Set Phase Code: `6`
5. Set Fill Volume: `12`
6. Set Calibration: `6:1.8 12:2.9 20:4.1`
7. Check "Hybrid mode" (already checked by default)
8. Check "Generate residual plots"
9. Click "🚀 Run Pipeline"

---

## Files Created:
- `gui_web.py` - Web-based GUI (Flask application)
- `gui_pipeline.py` - Desktop GUI (tkinter application)
- `README_GUI.md` - This documentation

## Dependencies:
- Flask (for web GUI): Already installed
- matplotlib (for plots): Should be installed for residual plots
- tkinter (for desktop GUI): Optional, not available in current environment
