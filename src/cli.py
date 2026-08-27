# src/cli.py
import argparse
import ast
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from .schema import CellSchema, TreeSchema, PortSchema
from .harvester import IntelligentHarvester

def cmd_harvest(args):
    """Harvest public APIs from a package and merge into trees/{domain}.json."""
    domain = args.domain or args.package
    package = args.package
    trees_dir = Path(args.trees_dir)
    trees_dir.mkdir(parents=True, exist_ok=True)
    out_file = trees_dir / f"{domain}.json"

    print(f"[*] Initializing Intelligent Harvester for package '{package}' (domain: '{domain}')...")
    harvester = IntelligentHarvester(domain=domain, package_name=package)
    cells = harvester.harvest_all()
    print(f"[+] Harvested {len(cells)} function cells from '{package}'.")

    harvester.merge_and_save(cells, out_file)
    print(f"[+] Merged and saved into '{out_file}'.")


def init_sqlite_db(db_path: Path) -> sqlite3.Connection:
    """Initializes standard NSTL SQLite schema."""
    if db_path.exists():
        os.remove(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
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
            source_priority      INTEGER DEFAULT 100
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_role ON nodes(node_role)")
    conn.commit()
    return conn


def cmd_compile(args):
    """Compiles all trees/*.json domain files into a target SQLite database."""
    trees_dir = Path(args.trees_dir)
    out_db = Path(args.output)
    domain_filter = args.domains

    json_files = sorted(trees_dir.glob("*.json"))
    if domain_filter:
        json_files = [f for f in json_files if f.stem in domain_filter or any(d in f.stem for d in domain_filter)]

    print(f"[*] Compiling {len(json_files)} domain JSON files from '{trees_dir}' into '{out_db}'...")
    conn = init_sqlite_db(out_db)
    cur = conn.cursor()

    total_compiled = 0
    stats: Dict[str, int] = {}

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Failed to read {jf.name}: {e}")
            continue

        if isinstance(data, dict) and "cells" in data:
            try:
                tree = TreeSchema(**data)
                cells = tree.cells
                domain = tree.domain
            except Exception as e:
                print(f"[!] Schema validation error in {jf.name}: {e}")
                cells = []
                domain = jf.stem
        elif isinstance(data, list):
            cells = [CellSchema(**c) for c in data if isinstance(c, dict) and "cell_id" in c]
            domain = jf.stem.replace("_tree", "").replace("_seeds", "")
        else:
            continue

        count = 0
        for cell in cells:
            cid = cell.cell_id.strip().upper()
            primary_in = cell.primary_input
            primary_out = cell.primary_output

            in_type = primary_in.type_name
            in_state = primary_in.state
            out_type = primary_out.type_name
            out_state = primary_out.state

            cfg_dict = {
                "inputs": {k: v.model_dump() for k, v in cell.inputs.items()},
                "outputs": {k: v.model_dump() for k, v in cell.outputs.items()}
            }
            cfg_json = json.dumps(cfg_dict)
            deps_json = json.dumps(cell.dependencies)
            kws_json = json.dumps(cell.keywords or cell.semantic_tags)
            verified_val = 1 if cell.source_priority <= 10 else 0

            # Priority check: lower source_priority = higher trust (1 = seed, 100 = auto)
            cur.execute("SELECT source_priority FROM nodes WHERE cell_id = ?", (cid,))
            row = cur.fetchone()
            if row and row[0] < cell.source_priority:
                continue

            cur.execute("""
                INSERT OR REPLACE INTO nodes
                (cell_id, domain_name, node_type, node_role, stage, keywords,
                 input_type, input_state, output_type, output_state, code,
                 dependencies, configuration_schema, verified, source_provenance, source_priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid,
                cell.domain_name or domain,
                cell.node_type or "function",
                cell.node_role or "function",
                cell.stage,
                kws_json,
                in_type,
                in_state,
                out_type,
                out_state,
                cell.code_template,
                deps_json,
                cfg_json,
                verified_val,
                jf.name,
                cell.source_priority
            ))
            count += 1
            total_compiled += 1

        stats[domain] = count
        print(f"  [+] Domain '{domain}': compiled {count} nodes ({jf.name})")

    conn.commit()
    conn.close()
    print(f"[*] Compilation Complete: {total_compiled} total verified nodes compiled into '{out_db}'.")


def cmd_validate(args):
    """Performs dry-run AST validation and integrity verification on all nodes in SQLite."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[!] Database file '{db_path}' does not exist!")
        sys.exit(1)

    print(f"[*] Validating SQLite Database '{db_path}'...")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT cell_id, domain_name, stage, code, input_type, output_type FROM nodes")
    rows = cur.fetchall()

    valid_count = 0
    failed_count = 0
    errors: List[str] = []

    for row in rows:
        cell_id, domain, stage, code, in_t, out_t = row
        if not code or not code.strip():
            failed_count += 1
            errors.append(f"{cell_id}: Empty code template")
            continue

        # Replace all {placeholders} with dummy variables for AST dry-run
        dummy_code = re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", "dummy_var", code)
        try:
            ast.parse(dummy_code)
            valid_count += 1
        except SyntaxError as e:
            failed_count += 1
            errors.append(f"{cell_id}: AST Syntax Error: {e}")

    conn.close()

    print(f"\n==================================================")
    print(f" VALIDATION RESULTS FOR: {db_path.name}")
    print(f"==================================================")
    print(f" Total Nodes Checked : {len(rows)}")
    print(f" Syntactically Valid : {valid_count}")
    print(f" Failed Nodes        : {failed_count}")
    print(f" Success Rate        : {(valid_count / len(rows) * 100):.2f}%" if rows else "0.00%")

    if errors:
        print("\n[!] Top Errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n✅ All nodes in database passed 100% AST dry-run validation!")


def main():
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="NSTL Toolchain CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # harvest
    p_harvest = subparsers.add_parser("harvest", help="Harvest API primitives into single-file domain JSON")
    p_harvest.add_argument("package", type=str, help="Python package name to harvest (e.g. cv2, pandas)")
    p_harvest.add_argument("--domain", type=str, default=None, help="Target domain name (defaults to package name)")
    p_harvest.add_argument("--trees-dir", type=str, default="trees", help="Directory for domain tree JSON files")
    p_harvest.set_defaults(func=cmd_harvest)

    # compile
    p_compile = subparsers.add_parser("compile", help="Compile single-file domain JSONs into SQLite database")
    p_compile.add_argument("--trees-dir", type=str, default="trees", help="Directory containing domain JSON files")
    p_compile.add_argument("--output", type=str, default="trees/lattice.db", help="Target SQLite DB path")
    p_compile.add_argument("--domains", nargs="*", default=None, help="Optional domain filter")
    p_compile.set_defaults(func=cmd_compile)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate AST syntax and schema of all nodes in SQLite")
    p_validate.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
