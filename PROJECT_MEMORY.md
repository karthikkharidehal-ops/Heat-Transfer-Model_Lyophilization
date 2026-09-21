# Project Memory

## Project Overview
LyoXL Digital Twin - Lyophilization process modeling and optimization system.

## Current Phase
**Phase 0**: Governance, memory, and control-code guardrails implementation.

## Key Decisions Log

### E-GOV-INIT-001 (Current)
- **Decision**: Implement Phase 0 governance framework
- **Rationale**: Ensure traceability of physics parameter changes and prevent closed-loop control code from entering the codebase
- **Impact**: BLOCKING - All commits must pass governance tests

### E-QC-STEADY-001
- **Decision**: Replace setpoint-based phase detection with measured signal variance
- **Rationale**: Recipe setpoints do not prove physical steady-state; actual measurements must be used
- **Impact**: BLOCKING

### E-GEOM-AREA-001
- **Decision**: Fix heat transfer contact area calculation to use full lateral surface area
- **Rationale**: Previous implementation used only base area, causing unphysical Kv inflation (~20,000 W/m²/K)
- **Impact**: RESOLVES_BLOCKER - Phase 0/Phase 1 blocked

## Architecture Constraints
1. **No Closed-Loop Control**: The digital twin is a modeling/analysis tool only. No code shall contain `write_recipe`, `set_setpoint`, `PLC`, or `OPC-UA` functionality.
2. **Physics Parameter Traceability**: Any modification to core physics parameters (Kv, Rp0, A1, A2) requires an event log entry.
3. **Geometry Changes Require Assumption Updates**: Modifications to geometry_constants.yaml must be accompanied by assumption register updates.

## Open Questions
- [ ] Calibration certificate locations for all sensors
- [ ] Uncertainty quantification methodology for derived parameters

## References
- EVENTS.md - Chronological event log
- ASSUMPTIONS_REGISTER.md - Documented assumptions and their status
- DATA_DICTIONARY.md - Column definitions and units
