"""
tools/compile_trees.py - Neuro-Symbolic Topological Lattice (NSTL)
Compiles harvested JSON tree files and stubs into the relational SQLite database `trees/lattice.db`.
"""

from __future__ import annotations
import glob
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.path.join(PROJECT_ROOT, "trees", "lattice.db")


def sanitize_type_name(t: str) -> str:
    """Normalizes type strings to standard lattice canonical types."""
    if not t:
        return "any"
    t_clean = str(t).strip()
    t_lower = t_clean.lower()

    if t_lower in ("any", "anyobject", "object", "*", "top", "unknown", ""):
        return "any"
    if t_lower in ("_callable", "callable"):
        return "any"
    if t_lower in ("_sequence", "sequence"):
        return "list"
    if t_lower in ("_mapping", "mapping"):
        return "dict"

    if "dataframe" in t_lower:
        return "DataFrame"
    if "series" in t_lower:
        return "Series"
    if "ndarray" in t_lower or "array" in t_lower:
        return "ndarray"
    if "mat" in t_lower or "image" in t_lower or "umat" in t_lower:
        return "Mat"
    if "str" in t_lower or "filepath" in t_lower or "path" in t_lower:
        return "str"
    if "int" in t_lower or "integer" in t_lower:
        return "int"
    if "float" in t_lower or "double" in t_lower:
        return "float"
    if "bool" in t_lower or "boolean" in t_lower:
        return "bool"
    if "dict" in t_lower or "graph" in t_lower:
        return "dict"
    if "list" in t_lower or "tuple" in t_lower:
        return "list"

    return t_clean


def determine_stage(cell_id: str, code: str, in_type: str, out_type: str, raw_stage: Any = None) -> int:
    """Accurately classifies node into Stage 1 (Source), Stage 2 (Transform), or Stage 3 (Sink)."""
    cid_lower = cell_id.lower()
    code_lower = (code or "").lower()

    # Stage 3: Sinks / Exporters / Writers
    sink_indicators = [
        "to_csv", "to_parquet", "to_json", "to_excel", "to_sql", "to_feather", "to_pickle",
        "imwrite", "savefig", "save", "export", "dump", "tofile", "write"
    ]
    if any(k in cid_lower for k in sink_indicators) or any(f".{k}(" in code_lower for k in sink_indicators):
        return 3
    if out_type in ("None", "NoneType", "filepath_written") and in_type not in ("any", "str"):
        return 3

    # Stage 1: Ingest / Readers / Loaders / Creators
    source_indicators = [
        "read_csv", "read_parquet", "read_json", "read_excel", "read_sql", "read_feather",
        "imread", "load", "from_", "create_", "zeros", "ones", "arange", "linspace"
    ]
    if any(k in cid_lower for k in source_indicators) or any(f"{k}(" in code_lower for k in source_indicators):
        return 1
    if in_type in ("str", "int", "None") and out_type in ("DataFrame", "ndarray", "Mat"):
        return 1

    if raw_stage is not None and isinstance(raw_stage, int) and raw_stage in (1, 2, 3):
        return raw_stage

    return 2


def init_db(db_file=DB_PATH) -> sqlite3.Connection:
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
            verified             INTEGER DEFAULT 0
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_role ON nodes(node_role)')
    conn.commit()
    return conn


def compile_file_to_db(json_filepath: str, conn: sqlite3.Connection):
    """Compiles a harvested JSON file into the SQLite lattice table."""
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return

    nodes = data.get("nodes", data.get("cells", data)) if isinstance(data, dict) else data
    if not isinstance(nodes, list):
        return

    cursor = conn.cursor()
    compiled_count = 0

    for cell in nodes:
        if not isinstance(cell, dict):
            continue

        cell_id = cell.get("cell_id", "").upper().strip()
        if not cell_id or cell_id.startswith("_"):
            continue

        # Filter out builtins exception noise during database compilation
        cell_id_lower = cell_id.lower()
        if "arithmeticerror" in cell_id_lower or "baseexception" in cell_id_lower or "add_note" in cell_id_lower or "lookuperror" in cell_id_lower:
            continue

        domain_name = cell.get("domain_name", cell.get("domain", "generic")).lower()
        node_type = cell.get("node_type", cell.get("type", "function"))
        node_role = cell.get("node_role", "macro" if node_type == "macro" else "function")
        keywords = json.dumps(cell.get("keywords", []))

        # Multi-port inputs resolution
        inputs = cell.get("inputs", {})
        if isinstance(inputs, dict) and inputs:
            first_port = next(iter(inputs.values()))
            if isinstance(first_port, dict):
                in_type = first_port.get("type_name", first_port.get("type", "any"))
                in_state = first_port.get("state", "raw")
            else:
                in_type = "any"
                in_state = "raw"
        elif isinstance(inputs, list) and inputs:
            in_type = inputs[0].get("type_name", inputs[0].get("type", "any"))
            in_state = inputs[0].get("state", "raw")
        else:
            in_type = "any"
            in_state = "raw"

        # Multi-port outputs resolution
        outputs = cell.get("outputs", {})
        if isinstance(outputs, dict) and outputs:
            first_out = next(iter(outputs.values()))
            if isinstance(first_out, dict):
                out_type = first_out.get("type_name", first_out.get("type", "any"))
                out_state = first_out.get("state", "computed")
            else:
                out_type = "any"
                out_state = "computed"
        elif isinstance(outputs, list) and outputs:
            out_type = outputs[0].get("type_name", outputs[0].get("type", "any"))
            out_state = outputs[0].get("state", "computed")
        else:
            out_type = "any"
            out_state = "computed"

        # Sanitize types (eliminates AnyObject, _Callable, etc.)
        in_type = sanitize_type_name(in_type)
        out_type = sanitize_type_name(out_type)

        code = cell.get("code_template", cell.get("code", ""))
        stage = determine_stage(cell_id, code, in_type, out_type, cell.get("stage"))

        deps = json.dumps(cell.get("dependencies", [f"import {domain_name}"]))
        config_schema = json.dumps(cell.get("parameters", cell.get("inputs", {})))
        verified_val = 1 if cell.get("verified") else 0

        cursor.execute('''
            INSERT OR REPLACE INTO nodes
            (cell_id, domain_name, node_type, node_role, stage, keywords, input_type, input_state, output_type, output_state, code, dependencies, configuration_schema, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (cell_id, domain_name, node_type, node_role, stage, keywords, in_type, in_state, out_type, out_state, code, deps, config_schema, verified_val))
        compiled_count += 1

    conn.commit()
    print(f"[+] Compiled {compiled_count} nodes from {os.path.basename(json_filepath)} -> SQLite DB.")


def main():
    trees_dir = os.path.join(PROJECT_ROOT, "trees")
    db_path = os.path.join(trees_dir, "lattice.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = init_db(db_path)

    # Priority order: Compile trees/*_tree.json first, then harvests (skipping skeletons and builtins)
    tree_files = sorted([f for f in glob.glob(os.path.join(trees_dir, "*_tree.json")) if "builtins" not in f])
    harvest_files = sorted([f for f in glob.glob(os.path.join(PROJECT_ROOT, "harvests", "*.json")) if "builtins" not in f and "skeleton" not in f])
    all_files = tree_files + harvest_files

    print(f"[*] Starting Compilation of {len(all_files)} tree JSON files to SQLite...")
    for hf in all_files:
        compile_file_to_db(hf, conn)

    conn.close()

    # Harvest clean core patterns into lattice.db
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "harvesting"))
    try:
        from pattern_harvester import harvest_core_patterns
        harvest_core_patterns(db_path)
    except Exception as e:
        print(f"[!] Warning on pattern_harvester: {e}")

    print(f"[*] Compilation Complete: '{db_path}' is clean, multi-ported, and ready.")


if __name__ == "__main__":
    main()
