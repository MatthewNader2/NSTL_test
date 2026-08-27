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


KNOWN_FILE_EXTS = {
    'csv', 'json', 'jpg', 'jpeg', 'png', 'bmp', 'txt', 'db', 'h5', 'hdf5',
    'pdf', 'md', 'npz', 'pkl', 'pickle', 'feather', 'orc', 'avro', 'yaml', 'yml', 'toml', 'ini', 'parquet'
}


def _validate_template(code_template: str, cell_id: str, node_role: str = "function") -> Tuple[bool, Optional[str]]:
    """
    Validates that a code template is syntactically valid Python when placeholders
    are replaced with dummy values. Also checks for hardcoded filenames/constants using AST.
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

    try:
        parsed_tree = ast.parse(test_code)
    except SyntaxError as e:
        return False, f"Syntax error after placeholder substitution: {e}"

    # AST Walk: inspect Attribute nodes for unquoted bare filenames (e.g. data.csv passed as argument)
    for parent in ast.walk(parsed_tree):
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.Attribute):
                # If child is the function being called (e.g. dist.pdf(x) or resp.json()), it's a valid method call
                if isinstance(parent, ast.Call) and parent.func is child:
                    continue
                if child.attr.lower() in KNOWN_FILE_EXTS:
                    if isinstance(child.value, ast.Name):
                        if child.value.id not in ("pd", "pandas", "np", "numpy", "cv2", "scipy", "sklearn", "plt", "matplotlib"):
                            return False, f"Bare unquoted filename detected as attribute argument: {child.value.id}.{child.attr}"

            # Check string constants for hardcoded filenames in templates (e.g. pd.read_csv("data.csv") without placeholder)
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                val = child.value.strip()
                if any(val.lower().endswith(f".{ext}") for ext in KNOWN_FILE_EXTS):
                    if not ("{" in val and "}" in val):
                        if "/" not in val and "\\" not in val and len(val.split(".")) == 2:
                            return False, f"Hardcoded string filename detected: '{val}' (must use a placeholder)"

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

    # 1. ALWAYS yield the master cell itself!
    regular = dict(cell)
    if "variants" in regular:
        del regular["variants"]
    if "code" in regular and "code_template" not in regular:
        regular["code_template"] = regular.pop("code")
    elif not regular.get("code_template"):
        regular["code_template"] = _extract_code_template(regular)

    if "params" in regular and isinstance(regular["params"], list):
        regular["parameters"] = _params_to_dict(regular["params"])
        del regular["params"]

    master_code = regular.get("code_template", "")
    master_placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", master_code))

    yield regular

    # 2. If variants exist, yield only structurally valid variants
    if node_type == "special_nested" and isinstance(variants, list) and variants:
        parent_id = cell.get("cell_id", "").upper().strip()
        parent_inputs = cell.get("inputs", {})
        parent_outputs = cell.get("outputs", {})
        parent_params = cell.get("params", [])
        parent_keywords = set(cell.get("keywords", []))
        parent_domain = cell.get("domain_name", cell.get("domain", "generic")).lower()

        # Crucial required placeholders that must not be stripped
        crucial_phs = master_placeholders & {"filename", "filepath", "src", "img", "image", "df", "data", "x", "y"}

        for variant in variants:
            if not isinstance(variant, dict):
                continue

            variant_id = variant.get("variant_id", "")
            if not variant_id:
                continue

            code = (
                variant.get("code_snippet", "")
                or variant.get("code", "")
                or master_code
            )

            variant_phs = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", code))
            # Reject bogus variant if it stripped all crucial placeholders from the function call
            if crucial_phs and not (variant_phs & crucial_phs):
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
                "code_template": code
            }

            cfg = _params_to_dict(parent_params)
            for k, v in variant.items():
                if k in ("variant_id", "code_snippet", "code", "keywords", "description",
                         "inputs", "outputs", "variants"):
                    continue
                cfg[k] = v
            new_cell["parameters"] = cfg

            yield new_cell


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
            code = _extract_code_template(cell)

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

            params = cell.get("params", cell.get("parameters", cell.get("inputs", {})))
            config_schema_dict = _params_to_dict(params) if isinstance(params, list) else params
            config_schema = json.dumps(config_schema_dict)

            in_type, in_state = _resolve_first_port(cell.get("inputs", {}))
            out_type, out_state = _resolve_first_port(cell.get("outputs", {}))

            in_type = sanitize_type_name(in_type)
            out_type = sanitize_type_name(out_type)

            stage = determine_stage(cell_id, code, in_type, out_type, cell.get("stage"))

            # Refine stage 1 loader ports: ensure input state is source_identifier
            if stage == 1:
                if in_type in ("str", "any"):
                    in_state = "source_identifier"

            # Refine stage 3 sink ports: ensure primary input reflects data object being written
            if stage == 3 and isinstance(config_schema_dict, dict):
                for p_name, p_meta in config_schema_dict.items():
                    p_type = sanitize_type_name(p_meta.get("type", p_meta.get("type_name", "")) if isinstance(p_meta, dict) else str(p_meta))
                    if p_type in ("Mat", "DataFrame", "ndarray", "Series", "dict", "list"):
                        in_type = p_type
                        in_state = "any"
                        break
                if any(k in cell_id.lower() for k in ("to_csv", "imwrite", "savefig", "to_parquet", "to_json")):
                    out_state = "filepath_written"

            deps = cell.get("dependencies", [])
            if not deps and domain_name and domain_name not in ("generic", "python_core", "core"):
                deps = [f"import {domain_name}"]
            deps_json = json.dumps(deps)

            keywords = json.dumps(cell.get("keywords", []))
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
