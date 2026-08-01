import ast
import json
import os
import sys
import sqlite3
import glob

# Import fix_node from our newly moved fix_trees.py
from fix_trees import fix_node

DB_PATH = os.path.join("trees", "lattice.db")

def init_db():
    os.makedirs("trees", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
    # Create indices for O(1) router graph traversals
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)')
    conn.commit()
    return conn

def compile_python_to_db_and_json(py_filepath, conn):
    basename = os.path.basename(py_filepath)
    if basename.startswith("qwen_"):
        domain_name = basename.replace("qwen_", "").replace(".py", "")
    elif basename.startswith("harvested_"):
        domain_name = basename.replace("harvested_", "").replace(".py", "")
    else:
        return

    # Standardize name for json output
    output_json_path = os.path.join("trees", "micro", f"{domain_name}_auto.json")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(py_filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError) as e:
        print(f"[-] Parse error in {py_filepath}: {e}")
        return
    
    cells = []
    cursor = conn.cursor()
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            cell_id = node.name.upper()
            
            # Extract docstring
            docstring = ast.get_docstring(node) or ""
            keywords = []
            node_type = "function" # Default
            description_lines = []
            
            for line in docstring.split('\n'):
                line_lower = line.lower().strip()
                if line_lower.startswith('keywords:'):
                    kw_str = line.split(':', 1)[1].strip()
                    keywords = [k.strip() for k in kw_str.split(',')]
                elif line_lower.startswith('node_type:'):
                    node_type = line.split(':', 1)[1].strip()
                elif line_lower and not line_lower.startswith('---'):
                    description_lines.append(line.strip())
                    
            description = " ".join(description_lines).strip()
            
            # Extract Inputs
            in_type = "any"
            in_state = "raw"
            if node.args.args:
                arg = node.args.args[0]
                in_state = arg.arg
                if getattr(arg, 'annotation', None):
                    if isinstance(arg.annotation, ast.Name):
                        in_type = arg.annotation.id
                    elif isinstance(arg.annotation, ast.Constant):
                        in_type = arg.annotation.value
            
            # Extract Outputs
            out_type = "any"
            out_state = "computed"
            if getattr(node, 'returns', None):
                ret_str = ""
                if isinstance(node.returns, ast.Constant):
                    ret_str = node.returns.value
                elif isinstance(node.returns, ast.Name):
                    ret_str = node.returns.id
                
                if '_' in ret_str:
                    parts = ret_str.split('_', 1)
                    out_type = parts[0]
                    out_state = parts[1]
                else:
                    out_type = ret_str
                    out_state = "computed"
            
            # Extract code body
            body_nodes = node.body[1:] if ast.get_docstring(node) else node.body
            code_lines = [ast.get_source_segment(source, n) for n in body_nodes]
            code = "\n".join(filter(None, code_lines))
            
            # Convert standard python variables to NSTL injection templates
            code = code.replace("output_var", "{output_var}")
            code = code.replace("input_var", "{input_var}")
            
            # Extract imports
            dependencies = []
            for bnode in body_nodes:
                if isinstance(bnode, ast.Import):
                    for alias in bnode.names:
                        dependencies.append(alias.name)
                elif isinstance(bnode, ast.ImportFrom):
                    if bnode.module:
                        dependencies.append(bnode.module)

            # 1. Add to JSON array
            cell = {
                "cell_id": cell_id,
                "type": "micro",
                "node_type": node_type,
                "stage": 1,
                "description": description,
                "keywords": keywords,
                "inputs": {
                    "type_name": in_type,
                    "state": in_state
                },
                "outputs": {
                    "type_name": out_type,
                    "state": out_state
                },
                "domain_implementations": {
                    "Python_Core": {
                        "code": code,
                        "dependencies": dependencies
                    }
                }
            }
            cell = fix_node(cell, domain_name)
            cells.append(cell)

            # Extract fixed variables for SQLite insertion
            fixed_keywords_json = json.dumps(cell.get("keywords", []))
            fixed_deps_json = json.dumps(cell.get("domain_implementations", {}).get("Python_Core", {}).get("dependencies", []))
            fixed_in_type = cell.get("inputs", {}).get("type_name", "any")
            fixed_in_state = cell.get("inputs", {}).get("state", "raw")
            fixed_out_type = cell.get("outputs", {}).get("type_name", "any")
            fixed_out_state = cell.get("outputs", {}).get("state", "computed")
            fixed_code = cell.get("domain_implementations", {}).get("Python_Core", {}).get("code", "")
            
            cursor.execute('''
                INSERT OR REPLACE INTO nodes 
                (cell_id, domain_name, node_type, stage, keywords, input_type, input_state, output_type, output_state, code, dependencies, configuration_schema)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (cell_id, domain_name, node_type, 1, fixed_keywords_json, fixed_in_type, fixed_in_state, fixed_out_type, fixed_out_state, fixed_code, fixed_deps_json, '[]'))
            
    conn.commit()
    
    # Write JSON export
    domain_json = {
        "domain_name": domain_name,
        "version": "1.0.0",
        "description": f"Auto-compiled from {os.path.basename(py_filepath)}",
        "cells": cells
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # We still dump JSON, but optionally minified or mildly formatted depending on need.
        # User requested human readable but optimized. indent=2 is human readable.
        json.dump(domain_json, f, indent=2)
    
    print(f"[+] Compiled {len(cells)} nodes for {domain_name} -> lattice.db & {output_json_path}")


if __name__ == "__main__":
    conn = init_db()
    
    # Compute harvests dir relative to project root (parent of tools/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    harvests_dir = os.environ.get("NSTL_HARVESTS_DIR", os.path.join(project_root, "harvests"))
    files = glob.glob(os.path.join(harvests_dir, "*.py"))
    
    # Clear out old JSON auto files to prevent dirty state
    micro_dir = os.path.join("trees", "micro")
    if os.path.exists(micro_dir):
        for f in glob.glob(os.path.join(micro_dir, "*_auto.json")):
            os.remove(f)
            
    print(f"[*] Starting Compilation of {len(files)} libraries to SQLite & JSON...")
    for f in files:
        # Skip the expansion files directly, as they were already merged into the main files!
        if "EXPANSION" in f:
            continue
        compile_python_to_db_and_json(f, conn)
        
    conn.close()
    print("[*] Compilation Complete. `lattice.db` is ready for the Router!")
