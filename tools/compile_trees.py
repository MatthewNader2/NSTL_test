"""
tools/compile_trees.py - Neuro-Symbolic Topological Lattice (NSTL)
Compiles consolidated single-file domain JSONs (trees/*.json) into SQLite database.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import ast
import re
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.cli import init_sqlite_db, cmd_compile
from src.schema import TreeSchema, CellSchema

DB_PATH = os.path.join(PROJECT_ROOT, "trees", "lattice.db")


def _validate_template(code: str, cell_id: str, node_role: str = "function") -> Tuple[bool, Optional[str]]:
    """Validates AST syntax and rejects unquoted/hardcoded filename constants."""
    if not code or not code.strip():
        return False, "Empty code template"

    # Check for bare unquoted filenames (e.g. data.csv, image.jpg)
    bare_file_match = re.search(r'(?<![\'"])\b([a-zA-Z0-9_\-]+\.(?:csv|parquet|json|xlsx|jpg|jpeg|png|wav))\b(?![\'"])', code)
    if bare_file_match:
        matched_str = bare_file_match.group(1)
        if f"'{matched_str}'" not in code and f'"{matched_str}"' not in code and f"{{{matched_str}}}" not in code:
            return False, f"Bare unquoted filename argument '{matched_str}'"

    # Check for hardcoded literal filename strings in read/write/loader functions
    if any(k in cell_id.lower() for k in ("read_", "to_", "imread", "imwrite", "load", "save", "test_cell")):
        hardcoded_match = re.search(r'[\'"]([a-zA-Z0-9_\-/]+\.(?:csv|parquet|json|xlsx|jpg|jpeg|png|wav))[\'"]', code)
        if hardcoded_match:
            return False, f"Hardcoded string filename '{hardcoded_match.group(1)}'"

    dummy_code = re.sub(r'\{[a-zA-Z_][a-zA-Z0-9_]*\}', 'dummy_var', code)
    try:
        ast.parse(dummy_code)
        return True, None
    except SyntaxError as e:
        return False, f"AST SyntaxError: {e}"


def compile_database(output_db: str = DB_PATH, domain_filter: Optional[List[str]] = None):
    trees_dir = PROJECT_ROOT / "trees"

    class Args:
        trees_dir = str(PROJECT_ROOT / "trees")
        output = output_db
        domains = domain_filter

    cmd_compile(Args())


def main():
    parser = argparse.ArgumentParser(description="NSTL Tree Compiler")
    parser.add_argument("--output", type=str, default=DB_PATH, help="Target SQLite DB path")
    parser.add_argument("--domains", nargs="*", default=None, help="Filter by specific domains (e.g. pandas cv2)")
    args = parser.parse_args()

    compile_database(output_db=args.output, domain_filter=args.domains)


if __name__ == "__main__":
    main()
