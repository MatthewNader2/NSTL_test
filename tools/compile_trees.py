import ast
import glob
import json
import os
import sqlite3
import sys

# Import fix_node from fix_trees.py
from fix_trees import fix_node

DB_PATH = os.path.join("trees", "lattice.db")


def init_db(db_file=DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            cell_id TEXT PRIMARY KEY,
            domain_name TEXT,
            node_type TEXT,
            stage INTEGER,
            keywords TEXT,
            input_type TEXT,
            input_state TEXT,
            output_type TEXT,
            output_state TEXT,
            code TEXT,
            dependencies TEXT,
            configuration_schema TEXT
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)')
    conn.commit()
    return conn


def compile_json_nodes_to_db(json_filepath, conn, filter_verified_only=False):
    """Compile structured/enriched node JSON files to SQLite lattice database."""
    basename = os.path.basename(json_filepath)
    domain_name = basename.replace("enriched_", "").replace("verified_", "").replace("structural_", "").replace(".json", "")

    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get("cells", data) if isinstance(data, dict) else data
    if not isinstance(nodes, list):
        return

    cursor = conn.cursor()
    compiled_count = 0

    for cell in nodes:
        if not isinstance(cell, dict):
            continue

        if filter_verified_only and cell.get("verified") is False:
            continue

        cell_id = cell.get("cell_id", "").upper()
        if not cell_id:
            continue

        keywords = cell.get("keywords", [])
        keywords_json = json.dumps(keywords)

        impl = cell.get("domain_implementations", {}).get("Python_Core", {})
        code = impl.get("code", "")
        deps = impl.get("dependencies", [domain_name])
        deps_json = json.dumps(deps)

        inputs = cell.get("inputs", [{}])
        if isinstance(inputs, list) and inputs:
            in_type = inputs[0].get("type", inputs[0].get("type_name", "ndarray"))
            in_state = inputs[0].get("state", "raw")
        elif isinstance(inputs, dict):
            in_type = inputs.get("type_name", inputs.get("type", "ndarray"))
            in_state = inputs.get("state", "raw")
        else:
            in_type = "ndarray"
            in_state = "raw"

        outputs = cell.get("outputs", [{}])
        if isinstance(outputs, list) and outputs:
            out_type = outputs[0].get("type", outputs[0].get("type_name", "ndarray"))
            out_state = outputs[0].get("state", "computed")
        elif isinstance(outputs, dict):
            out_type = outputs.get("type_name", outputs.get("type", "ndarray"))
            out_state = outputs.get("state", "computed")
        else:
            out_type = "ndarray"
            out_state = "computed"

        # Discard wildcards
        if in_type.lower() in ("any", "any_computed"):
            in_type = "ndarray"
        if out_type.lower() in ("any", "any_computed"):
            out_type = "ndarray"

        node_type = cell.get("node_type", "function")

        cursor.execute('''
            INSERT OR REPLACE INTO nodes 
            (cell_id, domain_name, node_type, stage, keywords, input_type, input_state, output_type, output_state, code, dependencies, configuration_schema)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (cell_id, domain_name, node_type, 1, keywords_json, in_type, in_state, out_type, out_state, code, deps_json, '[]'))
        compiled_count += 1

    conn.commit()
    print(f"[+] Compiled {compiled_count} Tier 1-4 nodes from {basename} -> SQLite lattice DB.")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    harvests_dir = os.environ.get("NSTL_HARVESTS_DIR", os.path.join(project_root, "harvests"))
    trees_dir = os.path.join(project_root, "trees")

    db_path = os.path.join(trees_dir, "lattice.db")
    nstl_db_path = os.path.join(trees_dir, "nstl_lattice.db")

    # Re-initialize DBs cleanly
    for p in (db_path, nstl_db_path):
        if os.path.exists(p):
            os.remove(p)

    conn = init_db(db_path)
    conn_nstl = init_db(nstl_db_path)

    json_files = glob.glob(os.path.join(harvests_dir, "enriched_*.json"))
    if not json_files:
        json_files = glob.glob(os.path.join(harvests_dir, "structural_*.json"))

    print(f"[*] Starting Compilation of {len(json_files)} library node sets to SQLite...")
    for jf in json_files:
        compile_json_nodes_to_db(jf, conn, filter_verified_only=False)
        compile_json_nodes_to_db(jf, conn_nstl, filter_verified_only=False)

    conn.close()
    conn_nstl.close()
    print("[*] Compilation Complete. `trees/lattice.db` and `trees/nstl_lattice.db` are ready!")


if __name__ == "__main__":
    main()
