"""
Regression guard: every fit_parameters_joint call site in the production entry
points must be wired for mass-balance closure, and every report builder must
print the five mandated mass-balance lines.

Scans run_pipeline.py, gui_pipeline.py, and gui_web.py with AST parsing (no
execution) and fails if any Call node to fit_parameters_joint lacks an
``endpoint_time_s`` keyword argument. Also fails if report_builders.py lacks
any of the five required output lines.

Event: E-FIX-MASSBAL-WIRE-002
Phase: Phase 1 identification
Gate impact: RESOLVES_BLOCKER
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CALL_SITE_FILES = [
    REPO_ROOT / "run_pipeline.py",
    REPO_ROOT / "gui_pipeline.py",
    REPO_ROOT / "gui_web.py",
]

REPORT_BUILDER_FILES = [
    REPO_ROOT / "report_builders.py",
]

REQUIRED_MASS_BALANCE_LINES = [
    "Detected endpoint time (s):",
    "Simulated endpoint time (s):",
    "Mass-balance residual:",
    "Mass-balance active:",
    "Mass-balance penalty share of cost:",
]


def _call_func_name(node: ast.Call) -> str:
    """Return the trailing function name for a Call node (Name or Attribute)."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _find_fit_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_func_name(node) == "fit_parameters_joint":
            calls.append(node)
    return calls


def test_all_fit_parameters_joint_calls_pass_endpoint_time_s():
    """Every fit_parameters_joint call in the pipeline entry points must pass
    endpoint_time_s as a keyword argument. Silent omission is forbidden."""
    violations = []
    checked = 0
    for path in CALL_SITE_FILES:
        assert path.exists(), f"Expected entry-point file missing: {path}"
        for call in _find_fit_calls(path):
            checked += 1
            kwarg_names = {kw.arg for kw in call.keywords if kw.arg is not None}
            if "endpoint_time_s" not in kwarg_names:
                violations.append(
                    f"{path.name}:{call.lineno}: fit_parameters_joint call "
                    f"is missing the endpoint_time_s keyword argument"
                )
    assert checked > 0, "No fit_parameters_joint call sites found — scanner is broken"
    assert not violations, "Unwired mass-balance call site(s):\n" + "\n".join(violations)


def test_report_builders_print_all_required_lines():
    """Every report builder module must emit all five mandated mass-balance
    lines so both GUIs and the CLI share one source of truth."""
    for path in REPORT_BUILDER_FILES:
        assert path.exists(), f"Expected report builder file missing: {path}"
        source = path.read_text(encoding="utf-8")
        # Collect all string constants in the module AST.
        literals = [
            node.value
            for node in ast.walk(ast.parse(source, filename=str(path)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        for required in REQUIRED_MASS_BALANCE_LINES:
            assert any(required in lit for lit in literals), (
                f"{path.name} is missing the required report line prefix "
                f"{required!r}"
            )


def test_gui_web_imports_shared_report_builder():
    """gui_web.py must not duplicate report logic; it must import
    build_mass_balance_lines from report_builders."""
    path = REPO_ROOT / "gui_web.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_from_report_builders = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "report_builders":
            imported_from_report_builders.update(alias.name for alias in node.names)
    assert "build_mass_balance_lines" in imported_from_report_builders, (
        "gui_web.py must import build_mass_balance_lines from report_builders "
        "(no duplicated report logic)"
    )


if __name__ == "__main__":
    test_all_fit_parameters_joint_calls_pass_endpoint_time_s()
    test_report_builders_print_all_required_lines()
    test_gui_web_imports_shared_report_builder()
    print("All call-site wiring regression tests passed!")
