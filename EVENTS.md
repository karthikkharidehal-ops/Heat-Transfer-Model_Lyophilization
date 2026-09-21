# Events Log

This document records all significant events, decisions, and changes to the LyoXL digital twin project.

## Event Format
- **Event ID**: Unique identifier (E-XXX-NNN)
- **Timestamp**: When the event occurred
- **Type**: Category of change
- **Description**: What happened
- **Rationale**: Why this change was made
- **Impact**: Effect on project phases/gates
- **Evidence**: Files or tests that verify the change

---

## Event Entries

### E-GOV-INIT-001
- **Timestamp**: 2024
- **Type**: Governance Initialization
- **Description**: Implement Phase 0 governance framework including event logging, assumptions register, and control-code guardrails
- **Rationale**: Ensure traceability of physics parameter changes and prevent closed-loop control code from entering the codebase
- **Impact**: BLOCKING - All subsequent commits must pass governance tests
- **Evidence**: PROJECT_MEMORY.md, tests/test_governance.py, src/governance/event_logger.py

**Note**: This event covers all existing physics parameter usages in the codebase at time of governance initialization:
- pikal_model.py: Contains Kv, Rp0, A1, A2 (Pikal model parameters)
- gui_web.py: Contains Rp0, A1, A2 (GUI display of resistance parameters)
- gui_pipeline.py: Contains Rp0, A1, A2 (GUI pipeline display)
- geometry.py: Contains geometry calculations
- run_pipeline.py: Contains pipeline orchestration

Future modifications to these files that change physics parameter behavior require new event entries.

### E-QC-STEADY-001
- **Timestamp**: 2024
- **Type**: Quality Control
- **Description**: Replace setpoint-based phase detection with measured signal variance for Hold/Ramp classification
- **Rationale**: Recipe setpoints do not prove physical steady-state; actual measurements must validate steady-state conditions
- **Impact**: BLOCKING
- **Evidence**: run_pipeline.py steady-state variance logic, DATA_DICTIONARY.md

### E-GEOM-AREA-001
- **Timestamp**: 2024
- **Type**: Physics Correction
- **Description**: Fix heat transfer contact area calculation to use full lateral surface area of frustum instead of base area only
- **Rationale**: Previous implementation artificially restricted heat transfer area, causing optimizer to inflate Kv to unphysical values (~20,000 W/m²/K)
- **Impact**: RESOLVES_BLOCKER - Phase 0/Phase 1 blocked
- **Evidence**: tests/test_physics_magnitudes.py

---

*This log is append-only. New events should be added at the bottom.*
