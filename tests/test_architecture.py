"""
Architecture enforcement tests for the lyophilization digital twin.

These tests ensure that GUI modules do not contain private copies of
canonical physics functions that should be imported from core_pipeline.py.

Event: E-ARCH-CORE-001
Phase: Phase 0
"""

import ast
from pathlib import Path


def get_function_definitions(source_code):
    """Extract all function and class definitions from source code."""
    tree = ast.parse(source_code)
    
    functions = []
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    
    return functions, classes


def test_no_private_build_segments_in_gui_pipeline():
    """FAIL if gui_pipeline.py contains a private definition of build_segments."""
    gui_path = Path(__file__).parent.parent / "gui_pipeline.py"
    source = gui_path.read_text(encoding="utf-8")
    functions, _ = get_function_definitions(source)
    
    assert "build_segments" not in functions, (
        "ARCHITECTURE VIOLATION: gui_pipeline.py contains a private definition of "
        "'build_segments'. This function must be imported from core_pipeline.py."
    )


def test_no_private_solve_Tp_in_gui_pipeline():
    """FAIL if gui_pipeline.py contains a private definition of solve_Tp."""
    gui_path = Path(__file__).parent.parent / "gui_pipeline.py"
    source = gui_path.read_text(encoding="utf-8")
    functions, _ = get_function_definitions(source)
    
    assert "solve_Tp" not in functions, (
        "ARCHITECTURE VIOLATION: gui_pipeline.py contains a private definition of "
        "'solve_Tp'. This function must be imported from the canonical module."
    )


def test_no_private_simulate_cycle_in_gui_pipeline():
    """FAIL if gui_pipeline.py contains a private definition of simulate_cycle."""
    gui_path = Path(__file__).parent.parent / "gui_pipeline.py"
    source = gui_path.read_text(encoding="utf-8")
    functions, _ = get_function_definitions(source)
    
    assert "simulate_cycle" not in functions, (
        "ARCHITECTURE VIOLATION: gui_pipeline.py contains a private definition of "
        "'simulate_cycle'. This function must be imported from pikal_model."
    )


def test_no_private_build_segments_in_gui_web():
    """FAIL if gui_web.py contains a private definition of build_segments."""
    gui_path = Path(__file__).parent.parent / "gui_web.py"
    source = gui_path.read_text(encoding="utf-8")
    functions, _ = get_function_definitions(source)
    
    assert "build_segments" not in functions, (
        "ARCHITECTURE VIOLATION: gui_web.py contains a private definition of "
        "'build_segments'. This function must be imported from core_pipeline.py."
    )


def test_no_private_solve_Tp_in_gui_web():
    """FAIL if gui_web.py contains a private definition of solve_Tp."""
    gui_path = Path(__file__).parent.parent / "gui_web.py"
    source = gui_path.read_text(encoding="utf-8")
    functions, _ = get_function_definitions(source)
    
    assert "solve_Tp" not in functions, (
        "ARCHITECTURE VIOLATION: gui_web.py contains a private definition of "
        "'solve_Tp'. This function must be imported from the canonical module."
    )


def test_no_private_simulate_cycle_in_gui_web():
    """FAIL if gui_web.py contains a private definition of simulate_cycle."""
    gui_path = Path(__file__).parent.parent / "gui_web.py"
    source = gui_path.read_text(encoding="utf-8")
    functions, _ = get_function_definitions(source)
    
    assert "simulate_cycle" not in functions, (
        "ARCHITECTURE VIOLATION: gui_web.py contains a private definition of "
        "'simulate_cycle'. This function must be imported from pikal_model."
    )


def test_core_pipeline_exists():
    """Verify that core_pipeline.py exists as the single source of truth."""
    core_path = Path(__file__).parent.parent / "core_pipeline.py"
    assert core_path.exists(), (
        "ARCHITECTURE VIOLATION: core_pipeline.py does not exist. "
        "This module is required as the single source of truth for pipeline physics."
    )


def test_core_pipeline_contains_canonical_functions():
    """Verify that core_pipeline.py contains all required canonical functions."""
    core_path = Path(__file__).parent.parent / "core_pipeline.py"
    source = core_path.read_text(encoding="utf-8")
    functions, classes = get_function_definitions(source)
    
    required_functions = [
        "build_segments",
        "choose_pressure_column",
        "choose_product_temperature_column",
        "coerce_phase_code",
        "parse_calib",
    ]
    
    required_classes = [
        "CalibratedPCRTubeGeometry",
    ]
    
    for func in required_functions:
        assert func in functions, (
            f"ARCHITECTURE VIOLATION: core_pipeline.py is missing required function '{func}'."
        )
    
    for cls in required_classes:
        assert cls in classes, (
            f"ARCHITECTURE VIOLATION: core_pipeline.py is missing required class '{cls}'."
        )


if __name__ == "__main__":
    import sys
    
    # Run all tests
    test_functions = [
        test_no_private_build_segments_in_gui_pipeline,
        test_no_private_solve_Tp_in_gui_pipeline,
        test_no_private_simulate_cycle_in_gui_pipeline,
        test_no_private_build_segments_in_gui_web,
        test_no_private_solve_Tp_in_gui_web,
        test_no_private_simulate_cycle_in_gui_web,
        test_core_pipeline_exists,
        test_core_pipeline_contains_canonical_functions,
    ]
    
    failed = False
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed = True
    
    sys.exit(1 if failed else 0)
