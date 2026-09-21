# Data Dictionary

This document describes all data columns used in the LyoXL digital twin pipeline, including their physical meaning, units, and calibration status.

## Pressure Measurements

| Column | Source | Units | Description | Calibration Uncertainty |
|--------|--------|-------|-------------|------------------------|
| `VACUUM` | Capacitance Manometer (Baratron) | mTorr | High-sensitivity chamber pressure reading from the capacitance manometer gauge. This is the primary pressure measurement used for physics modeling during primary drying. | REQUIRES_EVIDENCE |
| `capman_pa` | Derived from `VACUUM` | Pa | Capacitance manometer pressure converted to SI units (Pa). Conversion factor: 1 mTorr = 0.133322 Pa. | REQUIRES_EVIDENCE |
| `Pirani` | Pirani Gauge | mTorr | Chamber pressure reading from the Pirani gauge. This gauge is water-vapor dependent and reads differently than the capacitance manometer during primary drying due to thermal conductivity changes. Used primarily for convergence detection (end of primary drying). | REQUIRES_EVIDENCE |
| `pirani_pa` | Derived from `Pirani` | Pa | Pirani gauge pressure converted to SI units (Pa). | REQUIRES_EVIDENCE |
| `pressure_pa` | Selected column | Pa | The working pressure column used in the model. By default, this is `capman_pa` (preferred). Falls back to `pirani_pa`, `vacuum_pa`, or `vac_setpt_pa` if capacitance manometer data is unavailable. | N/A (derived) |

### Notes on Pressure Gauges

- **Capacitance Manometer (`VACUUM`)**: Measures absolute pressure via diaphragm deflection. Accurate for all gas compositions. This is the gold standard for chamber pressure measurement during lyophilization.

- **Pirani Gauge (`Pirani`)**: Measures pressure via thermal conductivity of the gas. Readings are gas-composition dependent. During primary drying, water vapor increases thermal conductivity, causing the Pirani to read higher than the capacitance manometer. The two gauges converge when water vapor is depleted (end of primary drying).

## Temperature Measurements

| Column | Source | Units | Description | Calibration Uncertainty |
|--------|--------|-------|-------------|------------------------|
| `shelf_temp_k` | Shelf RTD/Thermocouple | K | Measured shelf fluid temperature. This is the actual temperature reading from the shelf sensor, NOT the setpoint. Used in heat transfer calculations. | REQUIRES_EVIDENCE |
| `shelf_setpt_k` | PLC Setpoint | K | Shelf temperature setpoint commanded by the control system. **Not used for physics classification** — only for reference. May differ from actual `shelf_temp_k` during transients or if control valve is hunting. | N/A (setpoint) |
| `product_temp_k` | Product Thermocouples | K | Product temperature measurement. Either `prod_avg_k` (average of TP01-TP04) or `prod_min_k` (minimum reading, approximates ice-front temperature when probes are fully immersed). | REQUIRES_EVIDENCE |
| `prod_avg_k` | Average of TP01-TP04 | K | Arithmetic mean of all product temperature probes. | REQUIRES_EVIDENCE |
| `prod_min_k` | Minimum of TP01-TP04 | K | Minimum reading across all product temperature probes. Best approximation of ice-front temperature when thermocouples are fully immersed with metal parts exposed to different sections of the freezing front. | REQUIRES_EVIDENCE |
| `TPxx_k` | Individual Thermocouples | K | Raw product temperature readings from individual probes (e.g., TP01_k, TP02_k, etc.). | REQUIRES_EVIDENCE |

## Phase/Step Identification

| Column | Source | Units | Description |
|--------|--------|-------|-------------|
| `Phase` | Recipe/PLC | Integer/String | Phase identifier (e.g., Freezing=1, Primary Drying=6, Secondary Drying=7, Storage=8). User must supply the code for Primary Drying via `--primary-phase-code`. |
| `Cycle` | Historian | Integer | Cycle number within the batch. |
| `Step` | Recipe/PLC | Integer | Step number within the phase. |

## Derived Columns

| Column | Calculation | Units | Description |
|--------|-------------|-------|-------------|
| `timestamp` | Built from date/time columns | datetime | Unified timestamp for time-series alignment. |
| `converged` | `abs(pirani_pa - capman_pa) < tolerance` | Boolean | Flag indicating Pirani/capacitance manometer convergence (physically observed end of primary drying). |
| `vacuum_pa` | Alias for `capman_pa` | Pa | Legacy alias for capacitance manometer pressure. |

## Steady-State Detection Criteria

Hold vs Ramp step classification is now based on **measured signal variance**, not setpoint stability:

| Criterion | Threshold | Signal Used |
|-----------|-----------|-------------|
| Shelf temperature stability | std(`shelf_temp_k`) < 0.1 K | Measured shelf temperature |
| Chamber pressure stability | std(`capman_pa`) < 5.0 Pa | Capacitance manometer pressure |

A segment is classified as **"Hold" (steady-state)** ONLY if BOTH criteria are met. Otherwise, it is classified as **"Ramp" (transient)**.

---

**Evidence Requirements**: All columns marked `REQUIRES_EVIDENCE` must have calibration certificates or uncertainty analysis documentation before results can be used for regulatory submissions or critical process decisions.
