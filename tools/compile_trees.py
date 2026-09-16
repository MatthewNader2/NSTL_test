"""
tools/compile_trees.py - Neuro-Symbolic Topological Lattice (NSTL)
Compiles consolidated single-file domain JSONs (trees/*.json) into SQLite database.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Optional: only used if present; we no longer depend on it for compilation.
try:
    from src.cli import init_sqlite_db  # noqa: F401
except Exception:  # pragma: no cover - optional integration
    init_sqlite_db = None

DB_PATH = os.path.join(PROJECT_ROOT, "trees", "lattice.db")

# Schema — note the new `code_templates` column for polyglot templates.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS morphisms (
    cell_id TEXT PRIMARY KEY,
    domain TEXT,
    stage INTEGER,
    role TEXT,
    doc TEXT,
    inputs TEXT,
    outputs TEXT,
    template TEXT,
    code_templates TEXT
);
"""


def _validate_template(
    code: str, cell_id: str, node_role: str = "function"
) -> Tuple[bool, Optional[str]]:
    """Validates AST syntax and rejects unquoted/hardcoded filename constants."""
    if not code or not code.strip():
        return False, "Empty code template"

    # Check for bare unquoted filenames (e.g. data.csv, image.jpg)
    bare_file_match = re.search(
        r'(?<![\'"])\b([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]{1,8})\b(?![\'"])', code
    )
    if bare_file_match:
        matched_str = bare_file_match.group(1)
        if (
            f"'{matched_str}'" not in code
            and f'"{matched_str}"' not in code
            and f"{{{matched_str}}}" not in code
        ):
            return False, f"Bare unquoted filename argument '{matched_str}'"

    # Check for hardcoded literal filename strings in code templates
    hardcoded_match = re.search(
        r'[\'"]([a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]{1,8})[\'"]', code
    )
    if hardcoded_match:
        return False, f"Hardcoded string filename '{hardcoded_match.group(1)}'"

    dummy_code = re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", "dummy_var", code)
    try:
        ast.parse(dummy_code)
        return True, None
    except SyntaxError as e:
        return False, f"AST SyntaxError: {e}"


def extract_node_data(node: dict, domain: str) -> tuple:
    """Extract a `morphisms` row tuple from a tree node.

    Supports polyglot `code_templates` (e.g. {"python": ..., "r": ...})
    with backward compatibility for the legacy single-language `template`.
    """
    cell_id = node["cell_id"]
    stage = node["stage"]
    role = node.get("role", "transform")
    doc = node.get("doc", "")
    inputs_json = json.dumps(node.get("inputs", {}))
    outputs_json = json.dumps(node.get("outputs", {}))

    # HARDCODE REMOVAL: Polyglot template extraction with backward-compatibility
    code_templates = node.get("code_templates", {}) or {}
    if "python" in code_templates:
        py_template = code_templates["python"]
    else:
        py_template = node.get("template", "")

    # Persist the full polyglot map; fall back to python-only for legacy nodes.
    code_templates_json = json.dumps(
        code_templates if code_templates else {"python": py_template}
    )

    return (
        cell_id,
        domain,
        stage,
        role,
        doc,
        inputs_json,
        outputs_json,
        py_template,
        code_templates_json,
    )


def _iter_tree_nodes(path: Path) -> Iterable[Tuple[str, dict]]:
    """Yield (domain, node) pairs from a tree JSON file.

    Handles common consolidated shapes:
      - {"domain": "...", "cells": [...]}
      - {"domain": "...", "nodes": [...]}
      - [ {...}, {...} ]            (domain inferred from filename)
      - {...}                       (single-node document)
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        default_domain = path.stem
        for node in data:
            yield (node.get("domain", default_domain), node)
        return

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported tree JSON shape in {path}: {type(data).__name__}")

    domain = data.get("domain", path.stem)
    cells = data.get("cells") or data.get("nodes") or data.get("morphisms")
    if isinstance(cells, list):
        for node in cells:
            yield (node.get("domain", domain), node)
        return

    yield (domain, data)


def _resolve_tree_files(
    trees_dir: Path, domain_filter: Optional[List[str]]
) -> List[Path]:
    files = sorted(Path(p) for p in glob.glob(str(trees_dir / "*.json")))
    # Exclude non-tree JSON such as manifests.
    files = [p for p in files if p.name not in {"manifest.json", "index.json"}]
    if domain_filter:
        wanted = {d.lower() for d in domain_filter}
        files = [p for p in files if p.stem.lower() in wanted]
    return files


def compile_database(
    output_db: str = DB_PATH, domain_filter: Optional[List[str]] = None
) -> int:
    """Compile tree JSON files into the SQLite lattice.

    Returns the number of morphism rows written.
    """
    trees_dir = PROJECT_ROOT / "trees"
    tree_files = _resolve_tree_files(trees_dir, domain_filter)

    if not tree_files:
        print(
            f"[compile_trees] No tree files found in {trees_dir}"
            + (f" for domains={domain_filter}" if domain_filter else "")
        )
        return 0

    Path(output_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_db)
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)

        # Migrate older DBs that predate the polyglot `code_templates` column.
        existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(morphisms)")}
        if "code_templates" not in existing_cols:
            cur.execute("ALTER TABLE morphisms ADD COLUMN code_templates TEXT")
        conn.commit()

        insert_sql = """
            INSERT INTO morphisms (
                cell_id, domain, stage, role, doc,
                inputs, outputs, template, code_templates
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cell_id) DO UPDATE SET
                domain=excluded.domain,
                stage=excluded.stage,
                role=excluded.role,
                doc=excluded.doc,
                inputs=excluded.inputs,
                outputs=excluded.outputs,
                template=excluded.template,
                code_templates=excluded.code_templates
        """

        written = 0
        for path in tree_files:
            try:
                nodes = list(_iter_tree_nodes(path))
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"[compile_trees] SKIP {path.name}: {exc}", file=sys.stderr)
                continue

            for domain, node in nodes:
                if "cell_id" not in node or "stage" not in node:
                    print(
                        f"[compile_trees] SKIP node in {path.name}: missing cell_id/stage",
                        file=sys.stderr,
                    )
                    continue

                row = extract_node_data(node, domain)
                py_template = row[7]
                ok, err = _validate_template(
                    py_template, row[0], node.get("role", "transform")
                )
                if not ok:
                    print(
                        f"[compile_trees] WARN {path.name}::{row[0]}: {err}",
                        file=sys.stderr,
                    )

                cur.execute(insert_sql, row)
                written += 1

        conn.commit()
        print(
            f"[compile_trees] Compiled {written} morphism(s) from "
            f"{len(tree_files)} file(s) into {output_db}"
        )
        return written
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="NSTL Tree Compiler")
    parser.add_argument(
        "--output", type=str, default=DB_PATH, help="Target SQLite DB path"
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Filter by specific domains (e.g. pandas cv2)",
    )
    args = parser.parse_args()

    compile_database(output_db=args.output, domain_filter=args.domains)


if __name__ == "__main__":
    main()
