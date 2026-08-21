import ast
import glob
import json
import os
import sqlite3
import sys

# Import fix_node and sanitize_type from fix_trees.py
from fix_trees import fix_node, sanitize_type

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
            configuration_schema TEXT,
            verified INTEGER DEFAULT 0
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)')
    conn.commit()
    return conn


def compile_tree_file_to_db(json_filepath, conn, filter_verified_only=False):
    """Compile structured 1-file tree JSON to SQLite lattice database."""
    basename = os.path.basename(json_filepath)
    if not basename.endswith("_tree.json"):
        return

    domain_name = basename.replace("_tree.json", "")

    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get("nodes", data.get("cells", [])) if isinstance(data, dict) else data
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
            in_type = inputs[0].get("type", inputs[0].get("type_name", "AnyObject"))
            in_state = inputs[0].get("state", "raw")
        else:
            in_type = "AnyObject"
            in_state = "raw"

        outputs = cell.get("outputs", [{}])
        if isinstance(outputs, list) and outputs:
            out_type = outputs[0].get("type", outputs[0].get("type_name", "AnyObject"))
            out_state = outputs[0].get("state", "computed")
        else:
            out_type = "AnyObject"
            out_state = "computed"

        # Sanitize type names
        in_type = sanitize_type(in_type, cell_id, domain_name)
        out_type = sanitize_type(out_type, cell_id, domain_name)

        node_type = cell.get("node_type", "function")
        verified_val = 1 if cell.get("verified") is True else 0

        # Store variants schema if present (for Nested Special Nodes)
        variants = cell.get("variants", [])
        config_schema = json.dumps({"variants": variants}) if variants else '[]'

        cursor.execute('''
            INSERT OR REPLACE INTO nodes 
            (cell_id, domain_name, node_type, stage, keywords, input_type, input_state, output_type, output_state, code, dependencies, configuration_schema, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (cell_id, domain_name, node_type, 1, keywords_json, in_type, in_state, out_type, out_state, code, deps_json, config_schema, verified_val))
        compiled_count += 1

        # Also compile nested variant sub-nodes into lattice DB so router can match them directly
        for v in variants:
            v_id = f"{cell_id}_{v.get('variant_id', '')}".upper()
            v_keywords = keywords + v.get("keywords", [])
            v_code = v.get("code_snippet", code)
            cursor.execute('''
                INSERT OR REPLACE INTO nodes 
                (cell_id, domain_name, node_type, stage, keywords, input_type, input_state, output_type, output_state, code, dependencies, configuration_schema, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (v_id, domain_name, "special_variant", 1, json.dumps(v_keywords), in_type, in_state, out_type, out_state, v_code, deps_json, '[]', verified_val))
            compiled_count += 1

    conn.commit()
    print(f"[+] Compiled {compiled_count} nodes & variants from {basename} -> SQLite lattice DB.")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trees_dir = os.path.join(project_root, "trees")

    db_path = os.path.join(trees_dir, "lattice.db")
    nstl_db_path = os.path.join(trees_dir, "nstl_lattice.db")

    # Re-initialize DBs cleanly
    for p in (db_path, nstl_db_path):
        if os.path.exists(p):
            os.remove(p)

    conn = init_db(db_path)
    conn_nstl = init_db(nstl_db_path)

    tree_files = glob.glob(os.path.join(trees_dir, "*_tree.json"))

    print(f"[*] Starting Compilation of {len(tree_files)} 1-file tree JSONs to SQLite...")
    for tf in tree_files:
        compile_tree_file_to_db(tf, conn, filter_verified_only=False)
        compile_tree_file_to_db(tf, conn_nstl, filter_verified_only=False)

    conn.close()
    conn_nstl.close()

    # Harvest core algorithmic & control flow patterns into DB
    sys.path.insert(0, os.path.join(project_root, "harvesting"))
    try:
        from pattern_harvester import harvest_core_patterns
        harvest_core_patterns(db_path)
        harvest_core_patterns(nstl_db_path)
        print("[+] Harvested core algorithmic patterns into SQLite DBs.")
    except Exception as e:
        print(f"[-] Warning: Failed to run pattern harvester: {e}")

    print("[*] Compilation Complete. `trees/lattice.db` and `trees/nstl_lattice.db` are ready!")


if __name__ == "__main__":
    main()

