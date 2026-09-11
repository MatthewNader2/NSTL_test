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
    assert "PANDAS_READ_CSV" in res["path"][0]
    assert "code" in res
    assert "pandas.read_csv" in res["code"]
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
