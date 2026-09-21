# Assumptions Register

This document tracks all assumptions made in the LyoXL digital twin model, their status, and required evidence for validation.

## Assumption Format
- **ID**: Unique identifier (A-XXX-NNN)
- **Category**: Type of assumption
- **Description**: What is being assumed
- **Status**: Current validation state
- **Evidence Required**: What proof is needed
- **Related Files**: Files that depend on this assumption

---

## Registered Assumptions

### A-GEOM-001
- **Category**: Geometry
- **Description**: PCR tube geometry is accurately modeled as a frustum with specified dimensions
- **Status**: REQUIRES_EVIDENCE
- **Evidence Required**: Physical measurements of actual PCR tubes used in experiments
- **Related Files**: geometry.py, geometry_constants.yaml

### A-GEOM-002
- **Category**: Geometry
- **Description**: Heat transfer contact area includes lateral surface area of frustum from h=0 to fill_height, plus base area
- **Status**: IMPLEMENTED
- **Evidence Required**: Experimental validation of Kv values falling within physical bounds (1-500 W/m²/K)
- **Related Files**: geometry.py, pikal_model.py, tests/test_physics_magnitudes.py

### A-PHYS-001
- **Category**: Physics
- **Description**: Kv (vial heat transfer coefficient) must fall within physical bounds of 1.0 to 500.0 W/m²/K
- **Status**: ENFORCED_BY_TEST
- **Evidence Required**: Fitted Kv values from experimental data
- **Related Files**: tests/test_physics_magnitudes.py, pikal_model.py

### A-PHYS-002
- **Category**: Physics
- **Description**: Steady-state conditions are determined by measured signal variance, not setpoint stability
- **Status**: IMPLEMENTED
- **Evidence Required**: Variance analysis showing shelf_temp_k std < 0.1 K and capman_pa std < threshold during hold steps
- **Related Files**: run_pipeline.py, DATA_DICTIONARY.md

### A-SENSOR-001
- **Category**: Sensors
- **Description**: VACUUM column represents capacitance manometer readings in mTorr
- **Status**: REQUIRES_EVIDENCE
- **Evidence Required**: Calibration certificate for capacitance manometer gauge
- **Related Files**: ingest.py, DATA_DICTIONARY.md

### A-SENSOR-002
- **Category**: Sensors
- **Description**: Pirani column represents separate Pirani gauge readings in mTorr
- **Status**: REQUIRES_EVIDENCE
- **Evidence Required**: Calibration certificate for Pirani gauge
- **Related Files**: ingest.py, DATA_DICTIONARY.md

### A-SENSOR-003
- **Category**: Sensors
- **Description**: Temperature sensors (TP01-TP04, ShelfTemp) provide accurate readings in Celsius
- **Status**: REQUIRES_EVIDENCE
- **Evidence Required**: Calibration certificates for all temperature sensors
- **Related Files**: ingest.py, DATA_DICTIONARY.md

### A-DATA-001
- **Category**: Data Quality
- **Description**: Historian timestamp format is consistent and parseable
- **Status**: PARTIALLY_VALIDATED
- **Evidence Required**: Verification across multiple historian export formats
- **Related Files**: ingest.py

---

## Status Legend
- **REQUIRES_EVIDENCE**: Assumption documented but not yet validated
- **IMPLEMENTED**: Code changes made to address assumption
- **ENFORCED_BY_TEST**: Automated tests verify this assumption
- **PARTIALLY_VALIDATED**: Some evidence exists but full validation pending
- **VALIDATED**: Full evidence and validation complete
- **INVALIDATED**: Assumption proven false, requires correction

---

*This register must be updated whenever geometry_constants.yaml is modified or new physics assumptions are introduced.*
