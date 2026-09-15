import os
import sys
from pathlib import Path
import pytest
from rich.console import Console

# Add src to sys.path
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path = [p for p in sys.path if Path(p).resolve() != ROOT_DIR.resolve()]
sys.path.insert(0, str(SRC_DIR))

from cli import PipelineDebugger, NSTLInteractiveShell, build_parser


def test_cli_parser_debug_flags():
    """Verify that --debug and -d flags are registered across subparsers."""
    parser = build_parser()
    
    # Test top-level flag
    args = parser.parse_args(["--debug"])
    assert args.debug is True

    # Test run subcommand with --debug
    args = parser.parse_args(["run", "load csv data.csv", "--debug"])
    assert args.debug is True
    assert args.prompt == "load csv data.csv"

    # Test run subcommand with -d
    args = parser.parse_args(["run", "show head", "-d"])
    assert args.debug is True

    # Test shell subcommand with --debug
    args = parser.parse_args(["shell", "--debug"])
    assert args.debug is True


def test_shell_debug_toggle():
    """Verify interactive shell debug command toggling."""
    shell = NSTLInteractiveShell(
        db_path=str(Path(__file__).parent.parent / "trees" / "lattice.db"),
        initial_profile="0",
        interactive=False,
        debug=False
    )
    assert shell.debug is False
    assert "DEBUG" not in shell.prompt

    # Toggle on
    shell.do_debug("on")
    assert shell.debug is True
    assert "DEBUG" in shell.prompt

    # Toggle off
    shell.do_debug("off")
    assert shell.debug is False
    assert "DEBUG" not in shell.prompt

    # Toggle via set
    shell.do_set("debug on")
    assert shell.debug is True
    shell.do_set("debug off")
    assert shell.debug is False


def test_pipeline_debugger_valid_query():
    """Verify PipelineDebugger executes each layer and captures timing/output."""
    null_console = Console(file=open(os.devnull, "w"))
    shell = NSTLInteractiveShell(
        db_path=str(Path(__file__).parent.parent / "trees" / "lattice.db"),
        initial_profile="0",
        interactive=False,
        debug=True
    )
    debugger = PipelineDebugger(
        orchestrator=shell.orchestrator,
        router=shell.router,
        gate=shell.gate,
        sandbox=shell.sandbox,
        active_profile="0",
        console=null_console
    )

    res = debugger.run("load csv data.csv and show head", execute_sandbox=False)
    assert res is not None
    assert "prompt" in res
    assert "latency_ms" in res
    assert res["latency_ms"] > 0
    assert "path" in res
    assert len(res["path"]) >= 2
    assert "READ_CSV" in res["path"][0]
    assert "code" in res
    assert "read_csv" in res["code"].lower()
    assert res["synth_ms"] >= 0
    assert res["route_ms"] >= 0
    assert res["plan_ms"] >= 0


def test_pipeline_debugger_failing_query():
    """Verify PipelineDebugger gracefully diagnoses unsatisfiable queries without crashing."""
    null_console = Console(file=open(os.devnull, "w"))
    shell = NSTLInteractiveShell(
        db_path=str(Path(__file__).parent.parent / "trees" / "lattice.db"),
        initial_profile="0",
        interactive=False,
        debug=True
    )
    debugger = PipelineDebugger(
        orchestrator=shell.orchestrator,
        router=shell.router,
        gate=shell.gate,
        sandbox=shell.sandbox,
        active_profile="0",
        console=null_console
    )

    res = debugger.run("zzqqxx_impossible_prompt_12345", execute_sandbox=False)
    assert res is not None
    assert "latency_ms" in res
    assert len(res["path"]) == 0
    assert res["code"] == ""


def test_cli_parser_new_subcommands():
    """Verify that audit, macro, benchmark subcommands and route-method flags parse correctly."""
    parser = build_parser()

    # Test audit subcommand
    args = parser.parse_args(["audit", "--output", "audit_test.json"])
    assert args.command == "audit"
    assert args.output == "audit_test.json"

    # Test macro subcommand
    args = parser.parse_args(["macro", "PD_READ_CSV", "PD_DROPNA", "--id", "TEST_MACRO", "--domain", "pandas"])
    assert args.command == "macro"
    assert args.cells == ["PD_READ_CSV", "PD_DROPNA"]
    assert args.id == "TEST_MACRO"
    assert args.domain == "pandas"

    # Test benchmark subcommand
    args = parser.parse_args(["benchmark", "--type", "matrix"])
    assert args.command == "benchmark"
    assert args.type == "matrix"

    # Test run subcommand with route-method and no-lint
    args = parser.parse_args(["run", "load input.csv", "--route-method", "M1", "--no-lint"])
    assert args.command == "run"
    assert args.route_method == "M1"
    assert args.no_lint is True

    # Test shell subcommand with route-method
    args = parser.parse_args(["shell", "-m", "M3"])
    assert args.command == "shell"
    assert args.route_method == "M3"


def test_cli_audit_and_macro_execution(tmp_path):
    """Verify cmd_audit and cmd_macro execute cleanly from CLI invocation."""
    from cli import cmd_audit, cmd_macro
    import argparse

    db_file = Path(__file__).parent.parent / "trees" / "lattice.db"
    out_json = tmp_path / "audit_report.json"

    # 1. Audit execution
    audit_args = argparse.Namespace(
        db=str(db_file),
        trees_dir=str(Path(__file__).parent.parent / "trees"),
        output=str(out_json),
        debug=False
    )
    cmd_audit(audit_args)
    assert out_json.exists()
    import json
    with open(out_json, "r") as f:
        data = json.load(f)
    assert "total_cells" in data
    assert data["total_cells"] > 0
    assert "summary" in data

    # 2. Macro execution
    macro_args = argparse.Namespace(
        db=str(db_file),
        trees_dir=str(Path(__file__).parent.parent / "trees"),
        cells=["PD_READ_CSV", "PD_DROPNA"],
        id="MACRO_CLI_UNITTEST",
        domain="pandas",
        doc="CLI Test Macro",
        debug=False
    )
    cmd_macro(macro_args)


def test_shell_route_method_and_audit():
    """Verify interactive shell /method, /audit, and /macro dispatching."""
    shell = NSTLInteractiveShell(
        db_path=str(Path(__file__).parent.parent / "trees" / "lattice.db"),
        initial_profile="0",
        interactive=False,
        debug=False
    )
    # Test method switching
    assert shell.route_method in (None, "M0")
    shell.do_method("M1")
    assert shell.route_method == "M1"
    assert "M1" in shell.prompt

    shell.do_method("M3")
    assert shell.route_method == "M3"
    assert "M3" in shell.prompt

    # Test audit execution in shell
    shell.do_audit("")

    # Test macro execution in shell
    shell.do_macro("PD_READ_CSV PD_DROPNA --id MACRO_SHELL_TEST")
    assert "MACRO_SHELL_TEST" in shell.orchestrator.loaded_cells

