# Data Dictionary

This document describes all data columns used in the LyoXL digital twin pipeline, including their physical meaning, units, and calibration status.

## Raw Input Columns (from Historian CSV)

These are the column names expected in the raw input CSV files from the lyophilizer historian.

| Column | Physical Sensor Type | Units | Description | Calibration Uncertainty |
|--------|---------------------|-------|-------------|------------------------|
| `VACUUM` | Capacitance Manometer (Baratron) | mTorr | High-sensitivity chamber pressure reading. This is the primary pressure measurement used for physics modeling during primary drying. | REQUIRES_VERIFICATION |
| `Pirani` | Pirani Gauge | mTorr | Chamber pressure reading from the Pirani gauge. Water-vapor dependent; used for convergence detection. | REQUIRES_VERIFICATION |
| `ShelfTemp` | RTD/Thermocouple | °C | Measured shelf fluid temperature. | REQUIRES_VERIFICATION |
| `ShelfSetpt` | PLC Setpoint | °C | Shelf temperature setpoint commanded by control system. | N/A (setpoint) |
| `CondTemp` | RTD/Thermocouple | °C | Condenser temperature. | REQUIRES_VERIFICATION |
| `ProdAvg` | Calculated | °C | Average product temperature (may be pre-calculated by historian). | REQUIRES_VERIFICATION |
| `TP01`, `TP02`, `TP03`, `TP04` | Product Thermocouples | °C | Individual product temperature probes. | REQUIRES_VERIFICATION |
| `VacSetpt` | PLC Setpoint | mTorr | Vacuum setpoint. | N/A (setpoint) |
| `Date` | System Clock | DD-MM-YYYY | Date of recording. | N/A (timestamp) |
| `Time` | System Clock | HH:MM:SS.mmm | Time of recording. | N/A (timestamp) |
| `Cycle` | Historian Tag | Integer | Cycle number within batch. | N/A (identifier) |
| `Recipe` | Historian Tag | String | Recipe name/identifier. | N/A (identifier) |
| `Phase` | PLC Tag | Integer/String | Phase identifier (e.g., Freezing=1, Primary Drying=6). | REQUIRES_VERIFICATION |
| `Step` | PLC Tag | Integer | Step number within phase. | REQUIRES_VERIFICATION |

## Derived Columns

These columns are computed during the ingestion/cleaning process.

| Column | Source | Units | Description | Calibration Uncertainty |
|--------|--------|-------|-------------|------------------------|
| `capman_pa` | `VACUUM * 0.133322` | Pa | Capacitance manometer pressure in SI units. | REQUIRES_VERIFICATION |
| `pirani_pa` | `Pirani * 0.133322` | Pa | Pirani gauge pressure in SI units. | REQUIRES_VERIFICATION |
| `pressure_pa` | Selected from above | Pa | Working pressure column (defaults to `capman_pa`). | N/A (derived) |
| `vacuum_pa` | Alias for `capman_pa` | Pa | Legacy alias. | REQUIRES_VERIFICATION |
| `shelf_temp_k` | `ShelfTemp + 273.15` | K | Shelf temperature in Kelvin. | REQUIRES_VERIFICATION |
| `shelf_setpt_k` | `ShelfSetpt + 273.15` | K | Shelf setpoint in Kelvin. | N/A (setpoint) |
| `cond_temp_k` | `CondTemp + 273.15` | K | Condenser temperature in Kelvin. | REQUIRES_VERIFICATION |
| `prod_avg_k` | `ProdAvg + 273.15` or mean(TP01-TP04) | K | Average product temperature in Kelvin. | REQUIRES_VERIFICATION |
| `prod_min_k` | min(TP01_k, TP02_k, TP03_k, TP04_k) | K | Minimum product temperature (ice-front approximation). | REQUIRES_VERIFICATION |
| `tp01_k`, `tp02_k`, `tp03_k`, `tp04_k` | `TPxx + 273.15` | K | Individual probe temperatures in Kelvin. | REQUIRES_VERIFICATION |
| `vac_setpt_pa` | `VacSetpt * 0.133322` | Pa | Vacuum setpoint in SI units. | N/A (setpoint) |
| `timestamp` | Combined Date + Time | datetime | Unified timestamp for time-series alignment. | N/A (derived) |
| `converged` | `abs(pirani_pa - capman_pa) < tolerance` | Boolean | Pirani/capacitance manometer convergence flag. | N/A (derived) |
| `pd_end_candidate` | Convergence sustained for N minutes | Boolean | End of primary drying candidate marker. | N/A (derived) |

### Notes on Pressure Gauges

- **Capacitance Manometer (`VACUUM`)**: Measures absolute pressure via diaphragm deflection. Accurate for all gas compositions. This is the gold standard for chamber pressure measurement during lyophilization.

- **Pirani Gauge (`Pirani`)**: Measures pressure via thermal conductivity of the gas. Readings are gas-composition dependent. During primary drying, water vapor increases thermal conductivity, causing the Pirani to read higher than the capacitance manometer. The two gauges converge when water vapor is depleted (end of primary drying).

## Steady-State Detection Criteria

Hold vs Ramp step classification is based on **measured signal variance**, not setpoint stability:

| Criterion | Threshold | Signal Used |
|-----------|-----------|-------------|
| Shelf temperature stability | std(`shelf_temp_k`) < 0.1 K | Measured shelf temperature |
| Chamber pressure stability | std(`capman_pa`) < 5.0 Pa | Capacitance manometer pressure |

A segment is classified as **"Hold" (steady-state)** ONLY if BOTH criteria are met. Otherwise, it is classified as **"Ramp" (transient)**.

---

**Verification Requirements**: All columns marked `REQUIRES_VERIFICATION` must have calibration certificates or uncertainty analysis documentation before results can be used for regulatory submissions or critical process decisions.
