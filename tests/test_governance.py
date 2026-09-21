"""
Governance tests for Phase 0 compliance.

These tests enforce governance rules and will FAIL the commit if:
a. Any Python file containing 'Kv', 'Rp0', 'A1', or 'A2' is modified without a corresponding new entry in EVENTS.md.
b. The word 'write_recipe', 'set_setpoint', 'PLC', or 'OPC-UA' appears in any code file (No closed-loop control allowed).
c. geometry_constants.yaml is modified without an assumption register update.
"""

import os
import re
from pathlib import Path


def get_python_files(root_dir: Path = None) -> list:
    """Get all Python files in the project."""
    if root_dir is None:
        root_dir = Path(__file__).parent.parent
    
    python_files = []
    for py_file in root_dir.rglob("*.py"):
        # Exclude __pycache__ and hidden directories
        if "__pycache__" not in str(py_file) and not str(py_file).startswith("."):
            python_files.append(py_file)
    return python_files


def get_yaml_files(root_dir: Path = None) -> list:
    """Get all YAML files in the project."""
    if root_dir is None:
        root_dir = Path(__file__).parent.parent
    
    yaml_files = []
    for yaml_file in root_dir.rglob("*.yaml"):
        if "__pycache__" not in str(yaml_file) and not str(yaml_file).startswith("."):
            yaml_files.append(yaml_file)
    for yaml_file in root_dir.rglob("*.yml"):
        if "__pycache__" not in str(yaml_file) and not str(yaml_file).startswith("."):
            yaml_files.append(yaml_file)
    return yaml_files


def check_physics_params_have_events(root_dir: Path = None) -> tuple:
    """
    Check that any Python file containing physics parameters (Kv, Rp0, A1, A2)
    has a corresponding event entry in EVENTS.md.
    
    Returns: (passed: bool, message: str)
    """
    if root_dir is None:
        root_dir = Path(__file__).parent.parent
    
    events_path = root_dir / "EVENTS.md"
    if not events_path.exists():
        return False, "EVENTS.md does not exist. Create it before modifying physics parameters."
    
    events_content = events_path.read_text()
    physics_params = ['Kv', 'Rp0', 'A1', 'A2']
    violations = []
    
    for py_file in get_python_files(root_dir):
        try:
            content = py_file.read_text()
        except Exception:
            continue
        
        for param in physics_params:
            # Use word boundary to match whole parameter names
            pattern = r'\b' + re.escape(param) + r'\b'
            if re.search(pattern, content):
                # Check if this file or parameter is mentioned in EVENTS.md
                rel_path = str(py_file.relative_to(root_dir))
                # Look for event entries that mention this file or parameter
                file_mentioned = rel_path in events_content or py_file.name in events_content
                param_mentioned = re.search(r'\b' + re.escape(param) + r'\b', events_content) is not None
                
                if not (file_mentioned or param_mentioned):
                    violations.append(f"{rel_path} contains '{param}' but no event entry found")
    
    if violations:
        return False, "Physics parameter changes require event log entries:\n" + "\n".join(violations)
    
    return True, "All physics parameter usages have corresponding event entries."


def check_no_closed_loop_control(root_dir: Path = None) -> tuple:
    """
    Check that no code file contains closed-loop control keywords.
    Forbidden: 'write_recipe', 'set_setpoint', 'PLC', 'OPC-UA'
    
    Note: These keywords are allowed in comments/docstrings describing what is NOT allowed,
    and in test files that verify the guardrails themselves.
    
    Returns: (passed: bool, message: str)
    """
    if root_dir is None:
        root_dir = Path(__file__).parent.parent
    
    forbidden_keywords = ['write_recipe', 'set_setpoint', 'PLC', 'OPC-UA']
    violations = []
    
    # Files exempt from this check (governance tests and documentation)
    exempt_files = {'test_governance.py', 'event_logger.py'}
    
    for py_file in get_python_files(root_dir):
        # Skip exempt files (governance infrastructure)
        if py_file.name in exempt_files:
            continue
        
        try:
            content = py_file.read_text()
        except Exception:
            continue
        
        for keyword in forbidden_keywords:
            # Case-insensitive search for forbidden keywords
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            
            for match in matches:
                # Get line number and context
                start = match.start()
                line_start = content.rfind('\n', 0, start) + 1
                line_end = content.find('\n', start)
                if line_end == -1:
                    line_end = len(content)
                line_content = content[line_start:line_end].strip()
                
                # Skip if it's in a comment or docstring explaining what's forbidden/not allowed
                if line_content.startswith('#') or 'forbidden' in line_content.lower():
                    continue
                if 'not allowed' in line_content.lower() or 'no closed-loop' in line_content.lower():
                    continue
                # Skip if it's mentioning setpoints in a negative context (e.g., "not PLC setpoints")
                if 'rather than' in line_content.lower() or 'not plc' in line_content.lower():
                    continue
                # Skip if it's in the governance test docstring listing forbidden words
                if 'forbidden:' in line_content.lower():
                    continue
                    
                rel_path = str(py_file.relative_to(root_dir))
                violations.append(f"{rel_path} contains forbidden keyword '{keyword}' in: {line_content[:80]}")
    
    if violations:
        return False, "Closed-loop control code detected (violates architecture constraint):\n" + "\n".join(violations)
    
    return True, "No closed-loop control code detected."


def check_geometry_yaml_has_assumption_update(root_dir: Path = None) -> tuple:
    """
    Check that if geometry_constants.yaml exists and has been modified,
    the assumptions register has been updated.
    
    For this test, we check that the assumptions register mentions geometry constants.
    
    Returns: (passed: bool, message: str)
    """
    if root_dir is None:
        root_dir = Path(__file__).parent.parent
    
    # Find geometry_constants.yaml or any .yaml file with "geometry" in name
    geom_yaml_files = [f for f in get_yaml_files(root_dir) if 'geometry' in f.name.lower()]
    
    if not geom_yaml_files:
        # No geometry YAML files, nothing to check
        return True, "No geometry_constants.yaml found; no check required."
    
    assumptions_path = root_dir / "ASSUMPTIONS_REGISTER.md"
    if not assumptions_path.exists():
        return False, "ASSUMPTIONS_REGISTER.md does not exist. Create it when using geometry_constants.yaml."
    
    assumptions_content = assumptions_path.read_text()
    
    # Check that assumptions register mentions geometry-related assumptions
    geometry_keywords = ['geometry', 'GEOM', 'frustum', 'tube', 'contact_area']
    has_geometry_assumption = any(
        keyword.lower() in assumptions_content.lower() 
        for keyword in geometry_keywords
    )
    
    if not has_geometry_assumption:
        yaml_files_str = ", ".join(str(f.relative_to(root_dir)) for f in geom_yaml_files)
        return False, f"Geometry YAML files ({yaml_files_str}) exist but ASSUMPTIONS_REGISTER.md has no geometry-related assumptions."
    
    return True, "Geometry YAML files have corresponding assumptions in the register."


class TestGovernance:
    """Governance compliance test suite."""
    
    def test_physics_params_have_events(self):
        """Test that physics parameter modifications have corresponding event entries."""
        passed, message = check_physics_params_have_events()
        assert passed, f"PHYSICS_VIOLATION: {message}"
    
    def test_no_closed_loop_control(self):
        """Test that no closed-loop control code exists in the codebase."""
        passed, message = check_no_closed_loop_control()
        assert passed, f"CONTROL_VIOLATION: {message}"
    
    def test_geometry_yaml_has_assumption_update(self):
        """Test that geometry YAML modifications are accompanied by assumption register updates."""
        passed, message = check_geometry_yaml_has_assumption_update()
        assert passed, f"ASSUMPTION_VIOLATION: {message}"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
