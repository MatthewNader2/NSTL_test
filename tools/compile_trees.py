"""
tools/compile_trees.py - Neuro-Symbolic Topological Lattice (NSTL)
Compiles harvested JSON tree files into the relational SQLite database.
NOW WITH: template validation, provenance tracking, and priority-aware deduplication.
"""

from __future__ import annotations
import ast
import glob
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.path.join(PROJECT_ROOT, "trees", "lattice.db")

# Priority: higher number = higher trust. verified > enriched > structural > tree > auto
SOURCE_PRIORITY = {
    "verified": 4,
    "enriched": 3,
    "structural": 2,
    "tree": 1,
    "auto": 0,
    "seed": 2,
    "macro": 1,
    "llm": 0,
}

_TYPE_CANONICAL_MAP = {
    "any": "any", "anyobject": "any", "object": "any", "unknown": "any",
    "": "any", "*": "any", "top": "any",
    "_callable": "any", "callable": "any",
    "_sequence": "list", "sequence": "list",
    "_mapping": "dict", "mapping": "dict",
    "dataframe": "DataFrame", "pd.dataframe": "DataFrame",
    "series": "Series", "pd.series": "Series",
    "ndarray": "ndarray", "numpy.ndarray": "ndarray", "array": "ndarray",
    "mat": "Mat", "matlike": "Mat", "image": "Mat", "umat": "Mat",
    "cv2.typing.matlike": "Mat",
    "str": "str", "string": "str", "filepath": "str", "path": "str", "pathlike": "str",
    "int": "int", "integer": "int",
    "float": "float", "double": "float",
    "bool": "bool", "boolean": "bool",
    "dict": "dict", "dictionary": "dict", "graph": "dict",
    "list": "list", "tuple": "list", "sequence": "list",
    "nonetype": "None", "none": "None",
}


def _detect_source_priority(filepath: str) -> int:
    """Infer source trust level from filename."""
    name = os.path.basename(filepath).lower()
    for key, prio in SOURCE_PRIORITY.items():
        if key in name:
            return prio
    return 0


def sanitize_type_name(t: str) -> str:
    if not t:
        return "any"
    t_clean = str(t).strip()
    return _TYPE_CANONICAL_MAP.get(t_clean.lower(), t_clean)


def determine_stage(cell_id: str, code: str, in_type: str, out_type: str, raw_stage: Any = None) -> int:
    cid_lower = cell_id.lower()
    code_lower = (code or "").lower()

    sink_indicators = [
        "to_csv", "to_parquet", "to_json", "to_excel", "to_sql", "to_feather", "to_pickle",
        "imwrite", "savefig", "save", "export", "dump", "tofile", "write"
    ]
    if any(k in cid_lower for k in sink_indicators):
        return 3
    if code_lower:
        for k in sink_indicators:
            if f".{k}(" in code_lower:
                return 3
    if out_type in ("None", "NoneType", "filepath_written") and in_type not in ("any", "str"):
        return 3

    source_indicators = [
        "read_csv", "read_parquet", "read_json", "read_excel", "read_sql", "read_feather",
        "imread", "load", "from_", "create_", "zeros", "ones", "arange", "linspace"
    ]
    if any(k in cid_lower for k in source_indicators):
        return 1
    if code_lower:
        for k in source_indicators:
            if f"{k}(" in code_lower:
                return 1
    if in_type in ("str", "int", "None") and out_type in ("DataFrame", "ndarray", "Mat"):
        return 1

    if raw_stage is not None and isinstance(raw_stage, int) and raw_stage in (1, 2, 3):
        return raw_stage

    return 2


def _validate_template(code_template: str, cell_id: str, node_role: str = "function") -> Tuple[bool, Optional[str]]:
    """
    Validates that a code template is syntactically valid Python when placeholders
    are replaced with dummy values. Also checks for hardcoded filenames/constants.
    """
    if not code_template or not code_template.strip():
        # Macros legitimately have no code template; they use sub_cells
        if node_role == "macro":
            return True, None
        return False, "Empty template"

    # Find all placeholders like {var_name}
    placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", code_template))

    # Replace placeholders with safe dummy values
    test_code = code_template
    for ph in placeholders:
        test_code = test_code.replace(f"{{{ph}}}", "dummy_var")

    # Check for hardcoded filenames that are NOT module paths.
    # Module paths like pandas.io.json, pandas.io.parquet are OK.
    # We look for bare filenames without dots (or with just one extension dot).
    hardcoded = re.search(
        r"(?<![a-zA-Z0-9_\.])"           # not preceded by module path
        r"[a-zA-Z0-9_\-/]+"
        r"\.(csv|json|jpg|jpeg|png|bmp|txt|db|h5|hdf5|pdf|md|py|npz|pkl|pickle|feather|orc|avro|yaml|yml|toml|ini)"
        r"(?![a-zA-Z0-9_])",               # not followed by more path
        test_code,
        re.IGNORECASE
    )
    if hardcoded:
        match_str = hardcoded.group(0)
        # Reject only if it's not inside a string literal and not a module path
        if not (match_str.startswith(("'", '"')) and match_str.endswith(("'", '"'))):
            # Extra check: if it contains / or looks like a module path (has dots before it), allow
            if "/" not in match_str and ".." not in match_str:
                # But reject bare filenames like data.csv without quotes
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$", match_str):
                    return False, f"Hardcoded filename detected: {match_str}"

    try:
        ast.parse(test_code)
    except SyntaxError as e:
        return False, f"Syntax error after placeholder substitution: {e}"

    return True, None


def _extract_code_template(cell: Dict[str, Any]) -> str:
    return (
        cell.get("code_template", "")
        or cell.get("code", "")
        or cell.get("domain_implementations", {}).get("Python_Core", {}).get("code", "")
    )


def _params_to_dict(params: Any) -> Dict[str, Any]:
    if not isinstance(params, list):
        return params if isinstance(params, dict) else {}

    result = {}
    for p in params:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not name:
            continue
        entry = {"type": p.get("type", "any")}
        if "default" in p and p["default"] is not None and p["default"] != "...":
            entry["default_value"] = p["default"]
        if "required" in p:
            entry["required"] = p["required"]
        if "param_doc" in p:
            entry["doc"] = p["param_doc"]
        result[name] = entry
    return result


def _resolve_first_port(ports: Any) -> Tuple[str, str]:
    if isinstance(ports, dict) and ports:
        first = next(iter(ports.values()))
        if isinstance(first, dict):
            return (
                first.get("type_name", first.get("type", "any")),
                first.get("state", "raw")
            )
        return "any", "raw"
    elif isinstance(ports, list) and ports:
        first = ports[0]
        if isinstance(first, dict):
            return (
                first.get("type_name", first.get("type", "any")),
                first.get("state", "raw")
            )
        return "any", "raw"
    return "any", "raw"


def _iter_cells_to_compile(cell: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    node_type = str(cell.get("node_type", "")).lower()
    variants = cell.get("variants", [])

    if node_type == "special_nested" and isinstance(variants, list) and variants:
        parent_id = cell.get("cell_id", "").upper().strip()
        parent_inputs = cell.get("inputs", {})
        parent_outputs = cell.get("outputs", {})
        parent_params = cell.get("params", [])
        parent_keywords = set(cell.get("keywords", []))
        parent_domain = cell.get("domain_name", cell.get("domain", "generic")).lower()

        for variant in variants:
            if not isinstance(variant, dict):
                continue

            variant_id = variant.get("variant_id", "")
            if not variant_id:
                continue

            new_cell = {
                "cell_id": f"{parent_id}_{variant_id}".upper(),
                "domain_name": parent_domain,
                "node_type": "function",
                "node_role": "function",
                "inputs": variant.get("inputs", parent_inputs),
                "outputs": variant.get("outputs", parent_outputs),
                "keywords": list(parent_keywords | set(variant.get("keywords", []))),
                "dependencies": cell.get("dependencies", []),
                "stage": cell.get("stage"),
            }

            code = (
                variant.get("code_snippet", "")
                or variant.get("code", "")
                or _extract_code_template(cell)
            )
            new_cell["code_template"] = code

            cfg = _params_to_dict(parent_params)
            for k, v in variant.items():
                if k in ("variant_id", "code_snippet", "code", "keywords", "description",
                         "inputs", "outputs", "variants"):
                    continue
                cfg[k] = v
            new_cell["parameters"] = cfg

            yield new_cell
        return

    regular = dict(cell)
    if "code" in regular and "code_template" not in regular:
        regular["code_template"] = regular.pop("code")
    elif not regular.get("code_template"):
        regular["code_template"] = _extract_code_template(regular)

    if "params" in regular and isinstance(regular["params"], list):
        regular["parameters"] = _params_to_dict(regular["params"])
        del regular["params"]

    yield regular


def init_db(db_file: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            cell_id              TEXT PRIMARY KEY,
            domain_name          TEXT,
            node_type            TEXT,
            node_role            TEXT DEFAULT 'function',
            stage                INTEGER,
            keywords             TEXT,
            input_type           TEXT,
            input_state          TEXT,
            output_type          TEXT,
            output_state         TEXT,
            code                 TEXT,
            dependencies         TEXT,
            configuration_schema TEXT,
            verified             INTEGER DEFAULT 0,
            source_provenance    TEXT DEFAULT 'unknown',
            source_priority      INTEGER DEFAULT 0
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_role ON nodes(node_role)')
    conn.commit()
    return conn


def compile_file_to_db(json_filepath: str, conn: sqlite3.Connection):
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] Failed to load {json_filepath}: {e}")
        return

    nodes = data.get("nodes", data.get("cells", data)) if isinstance(data, dict) else data
    if not isinstance(nodes, list):
        print(f"[!] Unexpected structure in {json_filepath}")
        return

    file_priority = _detect_source_priority(json_filepath)
    cursor = conn.cursor()
    compiled_count = 0
    rejected_count = 0

    for raw_node in nodes:
        if not isinstance(raw_node, dict):
            continue

        for cell in _iter_cells_to_compile(raw_node):
            cell_id = cell.get("cell_id", "").upper().strip()
            if not cell_id or cell_id.startswith("_"):
                continue

            cid_lower = cell_id.lower()
            noise = ("arithmeticerror", "baseexception", "add_note", "lookuperror",
                     "exception", "warning", "error", "zombie")
            if any(p in cid_lower for p in noise):
                continue

            # === EXTRACT METADATA FIRST ===
            domain_name = cell.get("domain_name", cell.get("domain", "generic")).lower()
            node_type = cell.get("node_type", cell.get("type", "function"))
            node_role = cell.get("node_role", "macro" if node_type == "macro" else "function")

            code = cell.get("code_template", cell.get("code", ""))

            # === TEMPLATE VALIDATION (now node_role is defined) ===
            is_valid, reject_reason = _validate_template(code, cell_id, node_role=node_role)
            if not is_valid:
                cursor.execute("SELECT source_priority FROM nodes WHERE cell_id = ?", (cell_id,))
                row = cursor.fetchone()
                if row and row[0] >= file_priority:
                    continue
                print(f"  [!] REJECTED {cell_id} from {os.path.basename(json_filepath)}: {reject_reason}")
                rejected_count += 1
                continue

            keywords = json.dumps(cell.get("keywords", []))

            in_type, in_state = _resolve_first_port(cell.get("inputs", {}))
            out_type, out_state = _resolve_first_port(cell.get("outputs", {}))

            in_type = sanitize_type_name(in_type)
            out_type = sanitize_type_name(out_type)

            stage = determine_stage(cell_id, code, in_type, out_type, cell.get("stage"))

            deps = cell.get("dependencies", [])
            if not deps and domain_name and domain_name not in ("generic", "python_core", "core"):
                deps = [f"import {domain_name}"]
            deps_json = json.dumps(deps)

            parameters = cell.get("parameters", cell.get("inputs", {}))
            config_schema = json.dumps(parameters)

            verified_val = 1 if cell.get("verified") else 0

            cursor.execute("SELECT source_priority FROM nodes WHERE cell_id = ?", (cell_id,))
            existing = cursor.fetchone()
            if existing and existing[0] > file_priority:
                continue

            cursor.execute('''
                INSERT OR REPLACE INTO nodes
                (cell_id, domain_name, node_type, node_role, stage, keywords,
                 input_type, input_state, output_type, output_state, code,
                 dependencies, configuration_schema, verified, source_provenance, source_priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (cell_id, domain_name, node_type, node_role, stage, keywords,
                  in_type, in_state, out_type, out_state, code,
                  deps_json, config_schema, verified_val,
                  os.path.basename(json_filepath), file_priority))
            compiled_count += 1

    conn.commit()
    print(f"[+] Compiled {compiled_count} nodes from {os.path.basename(json_filepath)} -> SQLite DB. (rejected {rejected_count})")


def main():
    trees_dir = os.path.join(PROJECT_ROOT, "trees")
    db_path = os.path.join(trees_dir, "lattice.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = init_db(db_path)

    tree_files = sorted([
        f for f in glob.glob(os.path.join(trees_dir, "*_tree.json"))
        if "builtins" not in f
    ])
    harvest_files = sorted([
        f for f in glob.glob(os.path.join(PROJECT_ROOT, "harvests", "*.json"))
        if "builtins" not in f and "skeleton" not in f
    ])
    all_files = tree_files + harvest_files

    print(f"[*] Starting Compilation of {len(all_files)} tree JSON files to SQLite...")
    for hf in all_files:
        compile_file_to_db(hf, conn)

    conn.close()

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "harvesting"))
    try:
        from pattern_harvester import harvest_core_patterns
        harvest_core_patterns(db_path)
    except Exception as e:
        print(f"[!] Warning on pattern_harvester: {e}")

    print(f"[*] Compilation Complete: '{db_path}' is clean, multi-ported, and ready.")


if __name__ == "__main__":
    main()
