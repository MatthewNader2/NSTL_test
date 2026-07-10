import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATIC_PROMPT = 'Open "sample_input.txt" and strip whitespace from the text.'
CASES = [("A", "cpu"), ("A", "cuda"), ("B", "cpu"), ("B", "cuda")]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_code(code: str, output_dir: Path, case_name: str) -> dict:
    result = {
        "syntax_ok": False,
        "execution_ok": False,
        "syntax_error": None,
        "execution_returncode": None,
        "execution_stdout": "",
        "execution_stderr": "",
    }

    code_path = output_dir / f"{case_name}_generated.py"
    code_path.write_text(code, encoding="utf-8")

    try:
        ast.parse(code)
        result["syntax_ok"] = True
    except SyntaxError as exc:
        result["syntax_error"] = f"{exc.__class__.__name__}: {exc}"
        return result

    with tempfile.TemporaryDirectory(prefix=f"nstl_{case_name}_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "sample_input.txt").write_text("  hello lattice  \n", encoding="utf-8")
        run_path = tmp_path / "generated.py"
        run_path.write_text(code, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(run_path)],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=20,
        )
        result["execution_returncode"] = completed.returncode
        result["execution_stdout"] = completed.stdout[-4000:]
        result["execution_stderr"] = completed.stderr[-4000:]
        result["execution_ok"] = completed.returncode == 0

    return result


def run_case(profile: str, device: str, output_dir: Path) -> dict:
    case_name = f"profile_{profile}_{device}"
    started = time.perf_counter()
    payload = {
        "case": case_name,
        "profile": profile,
        "device": device,
        "prompt": STATIC_PROMPT,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "unknown",
        "timings": {},
        "initialize": None,
        "run": None,
        "validation": None,
        "error": None,
    }

    try:
        from fastapi.testclient import TestClient
        import main

        client = TestClient(main.app)

        t0 = time.perf_counter()
        init_response = client.post("/api/initialize", json={"profile": profile, "device": device})
        payload["timings"]["initialize_seconds"] = round(time.perf_counter() - t0, 4)
        payload["initialize"] = {
            "status_code": init_response.status_code,
            "json": init_response.json(),
        }
        if init_response.status_code >= 400 or init_response.json().get("status") == "error":
            payload["status"] = "initialize_failed"
            return payload

        t0 = time.perf_counter()
        run_response = client.post("/api/run", json={"prompt": STATIC_PROMPT})
        payload["timings"]["run_seconds"] = round(time.perf_counter() - t0, 4)
        run_json = run_response.json()
        payload["run"] = {
            "status_code": run_response.status_code,
            "logs": run_json.get("logs", []),
            "path_ids": [cell.get("cell_id") for cell in run_json.get("path", [])],
            "virtual_edges": run_json.get("virtual_edges", []),
            "code_length": len(run_json.get("code", "")),
        }

        code = run_json.get("code", "")
        payload["validation"] = _validate_code(code, output_dir, case_name)
        payload["status"] = "success" if payload["validation"]["syntax_ok"] else "generated_invalid_code"
        if not payload["validation"]["execution_ok"]:
            payload["status"] = "generated_code_execution_failed"
    except Exception as exc:
        payload["status"] = "exception"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
    finally:
        payload["timings"]["total_seconds"] = round(time.perf_counter() - started, 4)
        _write_json(output_dir / f"{case_name}.json", payload)

    return payload


def run_matrix(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for profile, device in CASES:
        case_name = f"profile_{profile}_{device}"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--profile",
            profile,
            "--device",
            device,
            "--output-dir",
            str(output_dir),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                timeout=900,
            )
            case_payload_path = output_dir / f"{case_name}.json"
            if case_payload_path.exists():
                case_payload = json.loads(case_payload_path.read_text(encoding="utf-8"))
            else:
                case_payload = {"case": case_name, "status": "missing_case_log"}
            case_payload["process"] = {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
                "wall_seconds": round(time.perf_counter() - started, 4),
            }
        except subprocess.TimeoutExpired as exc:
            case_payload = {
                "case": case_name,
                "profile": profile,
                "device": device,
                "status": "timeout",
                "process": {
                    "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                    "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                    "wall_seconds": round(time.perf_counter() - started, 4),
                },
            }
            _write_json(output_dir / f"{case_name}.json", case_payload)

        summary.append(case_payload)

    _write_json(
        output_dir / "summary.json",
        {
            "prompt": STATIC_PROMPT,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "cases": summary,
        },
    )

    for case in summary:
        validation = case.get("validation") or {}
        print(
            f"{case.get('case')}: {case.get('status')} "
            f"init={case.get('timings', {}).get('initialize_seconds')}s "
            f"run={case.get('timings', {}).get('run_seconds')}s "
            f"syntax={validation.get('syntax_ok')} exec={validation.get('execution_ok')}"
        )

    return 0


def main_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--profile", choices=["A", "B"])
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("benchmark_runs") / f"headless_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    if args.child:
        if not args.profile or not args.device:
            raise SystemExit("--child requires --profile and --device")
        payload = run_case(args.profile, args.device, output_dir)
        print(json.dumps({"case": payload["case"], "status": payload["status"]}))
        return 0

    return run_matrix(output_dir)


if __name__ == "__main__":
    raise SystemExit(main_cli())
