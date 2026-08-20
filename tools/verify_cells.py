"""
Tier 4 Runtime Verification Harness

Executes node code templates using synthetic input objects in an isolated scope.
Validates runtime execution and verifies observed vs declared output types.
Logs verification results to trees/verification_report.json and marks node verified flag.
"""

import ast
import json
import os
import sys
import traceback
from pathlib import Path

# Third-party imports for synthetic generation
import cv2
import numpy as np
import pandas as pd
import scipy
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_synthetic_value(param_name: str, param_type: str, default_val=None):
    """Generate a synthetic input stand-in object based on parameter name and type."""
    p_type_lower = param_type.lower() if param_type else ""

    if default_val is not None and default_val != "None" and default_val != "...":
        try:
            return eval(default_val, {"cv2": cv2, "np": np, "pd": pd})
        except Exception:
            pass

    if "mat" in p_type_lower or "ndarray" in p_type_lower:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    if "dataframe" in p_type_lower:
        return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    if "series" in p_type_lower:
        return pd.Series([1.0, 2.0, 3.0])
    if "int" in p_type_lower:
        return 1
    if "float" in p_type_lower or "double" in p_type_lower:
        return 1.0
    if "bool" in p_type_lower:
        return True
    if "str" in p_type_lower:
        return "synthetic_data.csv"
    if "tuple" in p_type_lower:
        return (1, 1)
    if "dict" in p_type_lower:
        return {"a": 1}
    if "list" in p_type_lower:
        return [1, 2, 3]
    if "graph" in p_type_lower:
        return {'A': {'B': 1}, 'B': {'A': 1}}
    
    return None


def verify_single_node(node: dict) -> dict:
    """Execute code template for a single node with synthetic args."""
    cell_id = node.get("cell_id", "UNKNOWN")
    impl = node.get("domain_implementations", {}).get("Python_Core", {})
    code_template = impl.get("code", "")
    declared_out_type = node.get("outputs", [{}])[0].get("type", "Any")

    if not code_template:
        return {"cell_id": cell_id, "verified": False, "error": "Empty code template"}

    # Prepare execution environment
    exec_env = {
        "cv2": cv2,
        "np": np,
        "pd": pd,
        "scipy": scipy,
        "sk": sklearn,
        "sklearn": sklearn,
        "output_var": None,
        "input_var": get_synthetic_value("input_var", node.get("inputs", [{}])[0].get("type", "Any"))
    }

    # Populate synthetic variable values for parameter placeholders
    substitutions = {
        "output_var": "output_var",
        "input_var": "input_var"
    }

    params = node.get("params", [])
    for p in params:
        p_name = p["name"]
        p_type = p.get("type", "Any")
        p_val = get_synthetic_value(p_name, p_type, p.get("default"))
        var_key = f"synthetic_{p_name}"
        exec_env[var_key] = p_val
        substitutions[p_name] = var_key

    # Format execution snippet
    snippet = code_template
    for placeholder, var_name in substitutions.items():
        snippet = snippet.replace(f"{{{placeholder}}}", var_name)

    try:
        exec(snippet, exec_env)
        result = exec_env.get("output_var")
        obs_type = type(result).__name__ if result is not None else "None"

        # Check type match heuristic
        type_match = False
        if declared_out_type in ("Mat", "ndarray") and isinstance(result, np.ndarray):
            type_match = True
        elif declared_out_type == "DataFrame" and isinstance(result, pd.DataFrame):
            type_match = True
        elif declared_out_type == "Series" and isinstance(result, pd.Series):
            type_match = True
        elif declared_out_type.lower() in (obs_type.lower(), "any", "object"):
            type_match = True

        return {
            "cell_id": cell_id,
            "verified": True,
            "observed_output_type": obs_type,
            "declared_output_type": declared_out_type,
            "type_match": type_match,
            "error": None
        }
    except Exception as e:
        return {
            "cell_id": cell_id,
            "verified": False,
            "observed_output_type": None,
            "declared_output_type": declared_out_type,
            "type_match": False,
            "error": str(e)
        }


def run_verification(input_json: str, output_verified_json: str, report_file: str = "trees/verification_report.json"):
    print(f"[*] Running Tier 4 Verification Harness on {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    report_entries = []
    verified_count = 0

    for node in nodes:
        res = verify_single_node(node)
        report_entries.append(res)

        if res["verified"]:
            node["verified"] = True
            verified_count += 1
        else:
            node["verified"] = False
            node["verification_error"] = res["error"]

    os.makedirs(os.path.dirname(os.path.abspath(report_file)), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_entries, f, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(output_verified_json)), exist_ok=True)
    with open(output_verified_json, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2)

    print(f"[+] Verification complete: {verified_count}/{len(nodes)} nodes verified.")
    print(f"    - Detailed report: {report_file}")
    print(f"    - Verified nodes saved: {output_verified_json}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_cells.py <input_nodes.json> <output_verified.json> [report_file.json]")
        sys.exit(1)

    rep = sys.argv[3] if len(sys.argv) > 3 else "trees/verification_report.json"
    run_verification(sys.argv[1], sys.argv[2], rep)
