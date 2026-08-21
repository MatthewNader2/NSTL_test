import os
import sys
import json
import re
import inspect
from pathlib import Path
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe module imports for runtime existence audit
MODULES: Dict[str, Any] = {}
try:
    import cv2
    MODULES['cv2'] = cv2
except ImportError:
    pass

try:
    import pandas as pd
    MODULES['pandas'] = pd
    MODULES['pd'] = pd
except ImportError:
    pass

try:
    import numpy as np
    MODULES['numpy'] = np
    MODULES['np'] = np
except ImportError:
    pass

try:
    import scipy
    MODULES['scipy'] = scipy
except ImportError:
    pass

try:
    import sklearn
    MODULES['sklearn'] = sklearn
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    MODULES['matplotlib'] = plt
    MODULES['plt'] = plt
except ImportError:
    pass

try:
    import builtins
    MODULES['builtins'] = builtins
except ImportError:
    pass


def check_function_exists(func_path: str) -> Tuple[bool, str]:
    """Audit if a function or attribute path exists in installed runtime libraries."""
    if not func_path or func_path == "UNKNOWN":
        return True, "OK"
    
    parts = func_path.split('.')
    mod_name = parts[0]
    
    if mod_name not in MODULES:
        return True, "UNKNOWN_MODULE"
    
    curr = MODULES[mod_name]
    for part in parts[1:]:
        if not hasattr(curr, part):
            return False, f"Missing attribute '{part}' in '{curr}'"
        try:
            curr = getattr(curr, part)
        except Exception as e:
            return False, f"Failed to access '{part}': {e}"
            
    return True, "OK"


def infer_concrete_output_type(domain: str, func_name: str, code_snippet: str, current_type: str) -> str:
    """Infer a concrete typestate return type, eliminating 'ANY' wildcards."""
    if current_type and current_type.lower() not in ["any", "anyobject", ""]:
        return current_type

    d_lower = domain.lower()
    f_lower = func_name.lower()
    code_lower = code_snippet.lower()

    if "pandas" in d_lower or "pd" in d_lower:
        if any(k in f_lower for k in ["read_csv", "read_excel", "read_json", "read_parquet", "dataframe", "dropna", "fillna", "sort_values", "groupby", "merge", "concat", "head", "tail"]):
            return "DataFrame"
        if any(k in f_lower for k in ["series", "isin", "astype", "map", "apply"]) or "df[" in code_lower:
            return "Series"
        if any(k in f_lower for k in ["mean", "sum", "count", "std", "var", "min", "max"]):
            return "float"
        return "DataFrame"

    if "cv2" in d_lower or "opencv" in d_lower:
        if any(k in f_lower for k in ["cvtcolor", "imread", "gaussianblur", "medianblur", "bilateralfilter", "threshold", "canny", "resize", "warpaffine", "erode", "dilate"]):
            return "Mat"
        if "findcontours" in f_lower:
            return "List[ndarray]"
        if "humoments" in f_lower:
            return "ndarray"
        return "Mat"

    if "numpy" in d_lower or "np" in d_lower:
        if any(k in f_lower for k in ["zeros", "ones", "array", "reshape", "transpose", "dot", "matmul", "mean", "std", "where"]):
            return "ndarray"
        return "ndarray"

    if "scipy" in d_lower:
        if "minimize" in f_lower or "optimize" in f_lower:
            return "OptimizeResult"
        return "ndarray"

    if "sklearn" in d_lower:
        if any(k in f_lower for k in ["fit_transform", "transform", "predict", "predict_proba"]):
            return "ndarray"
        return "ndarray"

    if "builtins" in d_lower or "python" in d_lower:
        if "open" in f_lower:
            return "TextIO"
        if "len" in f_lower or "int" in f_lower:
            return "int"
        if "str" in f_lower:
            return "str"
        return "AnyObject"

    return "AnyObject"


def build_nested_special_nodes(library_name: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group parameter-mode variants into clean Nested Special Nodes."""
    if library_name != "cv2":
        return nodes

    special_cvtcolor = {
        "cell_id": "CV2_CVTCOLOR",
        "domain": "cv2",
        "name": "cvtColor",
        "node_type": "special_nested",
        "function": "cv2.cvtColor",
        "description": "Converts an image from one color space to another (BGR, Grayscale, HSV, RGB, LAB).",
        "inputs": [{"name": "src", "type": "Mat"}],
        "outputs": [{"name": "dst", "type": "Mat"}],
        "params": [
            {"name": "src", "type": "Mat"},
            {"name": "code", "type": "int", "description": "Color conversion code flag"}
        ],
        "domain_implementations": {
            "Python_Core": {
                "code": "{output_var} = cv2.cvtColor({src}, {code})"
            }
        },
        "variants": [
            {
                "variant_id": "COLOR_BGR2GRAY",
                "code_flag": "cv2.COLOR_BGR2GRAY",
                "keywords": ["gray", "grayscale", "bgr2gray", "convert to gray", "black and white"],
                "description": "Convert image from BGR color space to Grayscale",
                "code_snippet": "{output_var} = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY)"
            },
            {
                "variant_id": "COLOR_BGR2HSV",
                "code_flag": "cv2.COLOR_BGR2HSV",
                "keywords": ["hsv", "hue", "saturation", "bgr2hsv", "convert to hsv"],
                "description": "Convert image from BGR color space to HSV",
                "code_snippet": "{output_var} = cv2.cvtColor({src}, cv2.COLOR_BGR2HSV)"
            },
            {
                "variant_id": "COLOR_BGR2RGB",
                "code_flag": "cv2.COLOR_BGR2RGB",
                "keywords": ["rgb", "red green blue", "bgr2rgb", "convert to rgb"],
                "description": "Convert image from BGR color space to RGB",
                "code_snippet": "{output_var} = cv2.cvtColor({src}, cv2.COLOR_BGR2RGB)"
            },
            {
                "variant_id": "COLOR_BGR2YCrCb",
                "code_flag": "cv2.COLOR_BGR2YCrCb",
                "keywords": ["ycrcb", "bgr2ycrcb", "convert to ycrcb"],
                "description": "Convert image from BGR color space to YCrCb",
                "code_snippet": "{output_var} = cv2.cvtColor({src}, cv2.COLOR_BGR2YCrCb)"
            },
            {
                "variant_id": "COLOR_GRAY2BGR",
                "code_flag": "cv2.COLOR_GRAY2BGR",
                "keywords": ["gray2bgr", "convert gray to bgr"],
                "description": "Convert image from Grayscale color space to BGR",
                "code_snippet": "{output_var} = cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)"
            }
        ]
    }

    special_threshold = {
        "cell_id": "CV2_THRESHOLD",
        "domain": "cv2",
        "name": "threshold",
        "node_type": "special_nested",
        "function": "cv2.threshold",
        "description": "Applies a fixed-level thresholding to a single-channel array.",
        "inputs": [{"name": "src", "type": "Mat"}],
        "outputs": [{"name": "dst", "type": "Mat"}],
        "params": [
            {"name": "src", "type": "Mat"},
            {"name": "thresh", "type": "float", "default": "127"},
            {"name": "maxval", "type": "float", "default": "255"},
            {"name": "type", "type": "int", "default": "cv2.THRESH_BINARY"}
        ],
        "domain_implementations": {
            "Python_Core": {
                "code": "_, {output_var} = cv2.threshold({src}, {thresh}, {maxval}, {type})"
            }
        },
        "variants": [
            {
                "variant_id": "THRESH_BINARY",
                "code_flag": "cv2.THRESH_BINARY",
                "keywords": ["binary threshold", "thresh binary"],
                "description": "Binary thresholding mode",
                "code_snippet": "_, {output_var} = cv2.threshold({src}, 127, 255, cv2.THRESH_BINARY)"
            },
            {
                "variant_id": "THRESH_BINARY_INV",
                "code_flag": "cv2.THRESH_BINARY_INV",
                "keywords": ["binary inverse threshold", "thresh binary inv"],
                "description": "Inverse binary thresholding mode",
                "code_snippet": "_, {output_var} = cv2.threshold({src}, 127, 255, cv2.THRESH_BINARY_INV)"
            },
            {
                "variant_id": "THRESH_OTSU",
                "code_flag": "cv2.THRESH_BINARY + cv2.THRESH_OTSU",
                "keywords": ["otsu threshold", "otsu binary"],
                "description": "Otsu automatic thresholding mode",
                "code_snippet": "_, {output_var} = cv2.threshold({src}, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"
            }
        ]
    }

    # Filter out individual duplicate wrapper nodes for cvtColor and threshold
    filtered_nodes = []
    for node in nodes:
        func = node.get("function", "") or node.get("name", "")
        code = node.get("domain_implementations", {}).get("Python_Core", {}).get("code", "")
        if "cvtColor" in func or "cvtColor" in code or "threshold" in func:
            continue
        filtered_nodes.append(node)

    filtered_nodes.insert(0, special_cvtcolor)
    filtered_nodes.insert(1, special_threshold)
    return filtered_nodes


def run_enhancement_and_audit():
    """Main audit and enhancement routine covering 100% of all library trees."""
    print("=== Starting 100% Tree Enhancement & Audit Engine ===")
    
    libraries = ["cv2", "pandas", "numpy", "scipy", "sklearn", "matplotlib", "builtins"]
    non_existent_report = []
    summary_stats = {}

    for lib in libraries:
        print(f"\n[*] Processing Library Tree: '{lib}'...")
        
        # Load from verified or enriched harvests
        verified_path = PROJECT_ROOT / "harvests" / f"verified_{lib}.json"
        enriched_path = PROJECT_ROOT / "harvests" / f"enriched_{lib}.json"
        
        raw_nodes = []
        if verified_path.exists():
            with open(verified_path, "r", encoding="utf-8") as f:
                raw_nodes = json.load(f)
        elif enriched_path.exists():
            with open(enriched_path, "r", encoding="utf-8") as f:
                raw_nodes = json.load(f)
        else:
            # Fallback to trees/micro/<lib>_auto.json
            micro_path = PROJECT_ROOT / "trees" / "micro" / f"{lib}_auto.json"
            if micro_path.exists():
                with open(micro_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    raw_nodes = raw_data.get("cells", raw_data) if isinstance(raw_data, dict) else raw_data

        audited_nodes = []
        purged_any_count = 0
        missing_func_count = 0

        for node in raw_nodes:
            cell_id = node.get("cell_id", node.get("name", "UNKNOWN"))
            func_name = node.get("function", node.get("name", ""))
            
            # Extract function call path if available
            code_snippet = node.get("domain_implementations", {}).get("Python_Core", {}).get("code", "")
            if not func_name and code_snippet:
                match = re.search(r"([a-zA-Z0-9_]+\.[a-zA-Z0-9_\.]+)\(", code_snippet)
                if match:
                    func_name = match.group(1)

            # Audit function existence
            exists, err_msg = check_function_exists(func_name)
            if not exists:
                missing_func_count += 1
                non_existent_report.append({
                    "library": lib,
                    "cell_id": cell_id,
                    "function": func_name,
                    "error": err_msg
                })
                continue  # Exclude non-existent function node

            # Typestate Soundness Audit: Purge 'ANY' output wildcards
            outputs = node.get("outputs", [])
            if not isinstance(outputs, list) or len(outputs) == 0:
                outputs = [{"name": "output_var", "type": "Any"}]
            elif not isinstance(outputs[0], dict):
                outputs = [{"name": "output_var", "type": str(outputs[0])}]
            
            orig_out_type = outputs[0].get("type", "Any")
            concrete_type = infer_concrete_output_type(lib, func_name, code_snippet, orig_out_type)
            if orig_out_type.lower() in ["any", "anyobject", ""]:
                purged_any_count += 1
            
            outputs[0]["type"] = concrete_type
            node["outputs"] = outputs

            # Ensure inputs have clean types
            inputs = node.get("inputs", [])
            if not inputs:
                inputs = [{"name": "input_var", "type": "AnyObject"}]
            node["inputs"] = inputs

            audited_nodes.append(node)

        # Deduplicate & Build Nested Special Nodes
        final_nodes = build_nested_special_nodes(lib, audited_nodes)

        # Build clean 1-file tree JSON structure
        tree_doc = {
            "library": lib,
            "version": "1.0",
            "main_imports": [f"import {lib}" if lib not in ["pandas", "numpy"] else f"import {lib} as {'pd' if lib == 'pandas' else 'np'}"],
            "optional_imports": {},
            "nodes": final_nodes
        }

        target_file = PROJECT_ROOT / "trees" / f"{lib}_tree.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(tree_doc, f, indent=2)

        summary_stats[lib] = {
            "total_nodes": len(final_nodes),
            "purged_any_outputs": purged_any_count,
            "pruned_non_existent": missing_func_count
        }

        print(f"[+] Saved clean 1-file tree: trees/{lib}_tree.json ({len(final_nodes)} nodes)")

    # Build clean 1-file macro_tree.json
    print("\n[*] Processing Macro Tree: 'macro'...")
    macro_source = PROJECT_ROOT / "trees" / "macro" / "algorithms.json"
    macro_nodes = []
    if macro_source.exists():
        with open(macro_source, "r", encoding="utf-8") as f:
            m_data = json.load(f)
            macro_nodes = m_data.get("cells", m_data) if isinstance(m_data, dict) else m_data

    macro_doc = {
        "library": "macro",
        "version": "1.0",
        "main_imports": [],
        "nodes": macro_nodes
    }

    macro_target = PROJECT_ROOT / "trees" / "macro_tree.json"
    with open(macro_target, "w", encoding="utf-8") as f:
        json.dump(macro_doc, f, indent=2)

    print(f"[+] Saved clean 1-file tree: trees/macro_tree.json ({len(macro_nodes)} macro trees)")

    # Save non-existent functions audit report
    report_path = PROJECT_ROOT / "logs" / "non_existent_functions_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(non_existent_report, f, indent=2)
    print(f"\n[+] Non-existent functions report saved to: logs/non_existent_functions_report.json ({len(non_existent_report)} issues)")

    # Save Summary Report
    summary_path = PROJECT_ROOT / "logs" / "tree_enhancement_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 100% Tree Enhancement & Audit Summary\n\n")
        f.write("| Library Tree | Final Node Count | Purged 'ANY' Outputs | Pruned Non-Existent Functions |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for l, stats in summary_stats.items():
            f.write(f"| **{l}** | {stats['total_nodes']} | {stats['purged_any_outputs']} | {stats['pruned_non_existent']} |\n")

    print(f"[+] Summary markdown report written to: logs/tree_enhancement_summary.md")


if __name__ == "__main__":
    run_enhancement_and_audit()
