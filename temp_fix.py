#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

trees_dir = Path("trees")
if not trees_dir.exists():
    raise FileNotFoundError("Directory 'trees' not found. Run this from repo root.")

# ---------------------------------------------------------
# 1. Domain-aware port resolution heuristics
# ---------------------------------------------------------
INPUT_PRIORITIES = [
    # Vision / Images
    "src", "src1", "img", "image", "mat", "img1", "array", "images",
    # Tabular / DataFrames
    "df", "data", "series", "left", "obj",
    # Numerical / Arrays
    "arr", "a", "x", "X", "m", "ary", "x1", "arrays",
    # Plotting / Canvas
    "ax", "fig", "plt",
    # Machine Learning
    "estimator", "model", "clf", "reg", "pipe", "pipeline",
    # Generators / File paths
    "filepath_or_buffer", "filepath", "filename", "path", "fp", "shape"
]

OUTPUT_PRIORITIES = [
    "output_var", "dst", "out", "result", "image", "df", "series",
    "arr", "X_trans", "y_pred", "ax", "fig", "estimator", "model"
]

def get_primary_in(node):
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict) or len(inputs) == 0:
        return "void"
    for name, p in inputs.items():
        if isinstance(p, dict) and p.get("port_role") == "primary":
            return name
    for p in INPUT_PRIORITIES:
        if p in inputs:
            return p
    return list(inputs.keys())[0]

def get_primary_out(node):
    outputs = node.get("outputs", {})
    if not isinstance(outputs, dict) or len(outputs) == 0:
        return "void"
    for name, p in outputs.items():
        if isinstance(p, dict) and p.get("port_role") == "primary":
            return name
    for p in OUTPUT_PRIORITIES:
        if p in outputs:
            return p
    return list(outputs.keys())[0]

# ---------------------------------------------------------
# 2. Repair JSON Trees
# ---------------------------------------------------------
print("[*] Repairing JSON domain trees...")
for fpath in sorted(trees_dir.rglob("*.json")):
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[!] Failed to parse {fpath}: {e}")
        continue

    is_dict_wrapper = isinstance(data, dict)
    nodes = data if isinstance(data, list) else data.get("nodes", data.get("cells", []))
    if isinstance(nodes, dict):
        nodes_list = list(nodes.values())
    else:
        nodes_list = nodes

    for node in nodes_list:
        cid = node.get("cell_id") or node.get("id") or ""

        # 1. Fix primary_in & primary_out
        node["primary_in"] = get_primary_in(node)
        node["primary_out"] = get_primary_out(node)

        # 2. Fix suspicious 'any' state in inputs
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict):
            for p_name, p_data in inputs.items():
                if isinstance(p_data, dict):
                    st = str(p_data.get("state", "")).strip().lower()
                    if st in {"any", "unknown", "none", "null", ""}:
                        if "IN_RANGE" in cid:
                            p_data["state"] = "HSV"
                        elif "MOMENTS" in cid:
                            p_data["state"] = "BINARY"
                        elif "CONVERT_SCALE_ABS" in cid:
                            p_data["state"] = "GRAY"
                        else:
                            p_data["state"] = "BGR"

                    # 3. Fix unspecialized 'object' type in pickle
                    tn = str(p_data.get("type_name", "")).strip().lower()
                    if tn == "object":
                        if "file" in p_name.lower():
                            p_data["type_name"] = "BinaryIO"
                        else:
                            p_data["type_name"] = "AnyData"

        # 4. Fix suspicious 'any' state in outputs
        outputs = node.get("outputs", {})
        if isinstance(outputs, dict):
            for p_name, p_data in outputs.items():
                if isinstance(p_data, dict):
                    st = str(p_data.get("state", "")).strip().lower()
                    if st in {"any", "unknown", "none", "null", ""}:
                        p_data["state"] = "BGR"

    fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  [+] Patched {fpath.name}")

# ---------------------------------------------------------
# 3. Repair SQLite Database (trees/lattice.db)
# ---------------------------------------------------------
db_path = trees_dir / "lattice.db"
if db_path.exists():
    print("[*] Repairing lattice.db schema and data...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name in ('cells', 'nodes');")
    tbl = cur.fetchone()
    if tbl:
        tbl_name = tbl[0]
        cur.execute(f"PRAGMA table_info({tbl_name});")
        cols = [c[1] for c in cur.fetchall()]

        # Add domain column if missing
        if "domain" not in cols:
            print("  [+] Adding missing 'domain' column to SQLite table...")
            cur.execute(f"ALTER TABLE {tbl_name} ADD COLUMN domain TEXT;")

        # Synchronize domain data from JSON trees
        for fpath in trees_dir.glob("*.json"):
            dom = fpath.stem
            try:
                tree_data = json.loads(fpath.read_text(encoding="utf-8"))
                tree_nodes = tree_data if isinstance(tree_data, list) else tree_data.get("nodes", tree_data.get("cells", []))
                if isinstance(tree_nodes, dict):
                    tree_nodes = tree_nodes.values()
                for nd in tree_nodes:
                    node_id = nd.get("cell_id") or nd.get("id")
                    if node_id:
                        cur.execute(f"UPDATE {tbl_name} SET domain = ? WHERE cell_id = ?", (dom, node_id))
            except Exception:
                pass
        conn.commit()
    conn.close()
    print("  [+] SQLite lattice.db successfully synchronized.")

print("\n[*] All trees and database records repaired successfully!")
