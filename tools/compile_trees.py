"""
tools/compile_trees.py - Neuro-Symbolic Topological Lattice (NSTL)
Compiles trees/*.json into SQLite.

Table and column names match the loader contract in src/lattice.py
(see the SELECT FROM nodes at ~line 1650):

    cell_id, domain_name, node_type, node_role, stage, keywords,
    input_type, input_state, output_type, output_state, code,
    dependencies, configuration_schema, slots, verified, docstring,
    source_priority

The loader's extended-state fields (inputs, outputs, slots, bound_slots,
postconditions, ...) are packed as a JSON dict into ``configuration_schema``.
Additional columns are kept as pass-through for downstream tooling; the
loader ignores unknown columns.
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
from typing import Any, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DB_PATH = os.path.join(PROJECT_ROOT, "trees", "lattice.db")


# --------------------------------------------------------------------------- #
# Column layout — matches the loader's SELECT FROM nodes field order.
# --------------------------------------------------------------------------- #

# Required by the loader (or synthesized by it).
_LOADER_COLUMNS: Tuple[str, ...] = (
    "cell_id",
    "domain_name",
    "node_type",
    "node_role",
    "stage",
    "keywords",
    "input_type",
    "input_state",
    "output_type",
    "output_state",
    "code",
    "dependencies",
    "configuration_schema",
    "slots",
    "verified",
    "docstring",
    "source_priority",
)

# Extra pass-through columns for downstream tooling; loader ignores them.
_EXTRA_COLUMNS: Tuple[str, ...] = (
    "domain",
    "doc",
    "inputs",
    "outputs",
    "code_templates",
    "topology_type",
    "mutation_type",
    "is_context_manager",
    "bound_slots",
    "preconditions",
    "postconditions",
    "effects",
    "type_vars",
    "raises",
    "semantic_tags",
    "primary_in",
    "primary_out",
    "is_public",
    "source_provenance",
)

_COLUMNS: Tuple[str, ...] = _LOADER_COLUMNS + _EXTRA_COLUMNS

_INT_COLUMNS: frozenset = frozenset({
    "stage", "verified", "is_context_manager", "is_public", "source_priority",
})

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    cell_id               TEXT PRIMARY KEY,
    domain_name           TEXT,
    node_type             TEXT,
    node_role             TEXT,
    stage                 INTEGER,
    keywords              TEXT,
    input_type            TEXT,
    input_state           TEXT,
    output_type           TEXT,
    output_state          TEXT,
    code                  TEXT,
    dependencies          TEXT,
    configuration_schema  TEXT,
    slots                 TEXT,
    verified              INTEGER,
    docstring             TEXT,
    source_priority       INTEGER,
    domain                TEXT,
    doc                   TEXT,
    inputs                TEXT,
    outputs               TEXT,
    code_templates        TEXT,
    topology_type         TEXT,
    mutation_type         TEXT,
    is_context_manager    INTEGER,
    bound_slots           TEXT,
    preconditions         TEXT,
    postconditions        TEXT,
    effects               TEXT,
    type_vars             TEXT,
    raises                TEXT,
    semantic_tags         TEXT,
    primary_in            TEXT,
    primary_out           TEXT,
    is_public             INTEGER,
    source_provenance     TEXT
);
"""

INSERT_SQL = f"""
    INSERT INTO nodes ({", ".join(_COLUMNS)})
    VALUES ({", ".join("?" * len(_COLUMNS))})
    ON CONFLICT(cell_id) DO UPDATE SET
        {", ".join(f"{c}=excluded.{c}" for c in _COLUMNS if c != "cell_id")}
"""


# --------------------------------------------------------------------------- #
# Structural helpers — no domain vocabulary, no extension lists.
# --------------------------------------------------------------------------- #

_DOTTED_NAME_RE = re.compile(
    r'(?<![\'"\w.])([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b(?![\'"])'
)
_QUOTED_PATH_RE = re.compile(r'(["\'])([^"\']*[/\\][^"\']*)\1')
_PLACEHOLDER_SUBST_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def _module_aliases(dependencies: Any) -> Set[str]:
    aliases: Set[str] = set()

    if isinstance(dependencies, str):
        dependencies = [dependencies]
    if not dependencies:
        return aliases

    def _add_dotted(path: str) -> None:
        aliases.add(path)
        aliases.add(path.split(".")[0])
        tail = path.rsplit(".", 1)[-1]
        if tail != path:
            aliases.add(tail)

    for dep in dependencies:
        if isinstance(dep, dict):
            for key in ("import", "module", "name", "statement"):
                if isinstance(dep.get(key), str):
                    dep = dep[key]
                    break
            else:
                for v in dep.values():
                    if isinstance(v, str):
                        dep = v
                        break
                else:
                    continue

        dep_s = str(dep).strip()
        if not dep_s:
            continue

        m = re.match(r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?", dep_s)
        if m:
            _add_dotted(m.group(1))
            if m.group(2):
                aliases.add(m.group(2))
            continue

        m = re.match(r"^from\s+([\w.]+)\s+import\s+(.+)$", dep_s)
        if m:
            _add_dotted(m.group(1))
            for name in m.group(2).split(","):
                name = name.strip()
                if not name or name == "*":
                    continue
                as_m = re.match(r"^(\w+)\s+as\s+(\w+)$", name)
                aliases.add(as_m.group(2) if as_m else name)
            continue

        m = re.match(r"^([\w.]+)(?:\s+as\s+(\w+))?$", dep_s)
        if m:
            _add_dotted(m.group(1))
            if m.group(2):
                aliases.add(m.group(2))
            continue

    return aliases


def _unbound_dotted_reference(token: str, declared_aliases: Set[str]) -> Optional[str]:
    if "." not in token:
        return None
    prefix = token.split(".", 1)[0]
    return None if prefix in declared_aliases else prefix


def _validate_template(
    code: str,
    cell_id: str,
    node_role: str = "function",
    declared_aliases: Optional[Set[str]] = None,
) -> Tuple[bool, Optional[str]]:
    if not code or not code.strip():
        return False, "Empty code template"

    aliases = declared_aliases or set()

    for m in _DOTTED_NAME_RE.finditer(code):
        token = m.group(1)
        if f"{{{token}}}" in code:
            continue
        unbound = _unbound_dotted_reference(token, aliases)
        if unbound is not None:
            return False, (
                f"Unbound reference '{token}' "
                f"(prefix '{unbound}' not in declared dependencies)"
            )

    for m in _QUOTED_PATH_RE.finditer(code):
        return False, f"Hardcoded path literal '{m.group(2)}'"

    dummy_code = _PLACEHOLDER_SUBST_RE.sub("dummy_var", code)
    try:
        ast.parse(dummy_code)
        return True, None
    except SyntaxError as e:
        return False, f"AST SyntaxError: {e}"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _json_or_null(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return json.dumps(v)
    except (TypeError, ValueError):
        return None


def _extract_python_template(node: dict) -> Tuple[str, dict]:
    code_templates = node.get("code_templates", {}) or {}
    if not isinstance(code_templates, dict):
        code_templates = {}
    py_template = ""
    if code_templates.get("python"):
        py_template = code_templates["python"]
    elif node.get("code_template"):
        py_template = node["code_template"]
    elif node.get("template"):
        py_template = node["template"]
    if py_template and "python" not in code_templates:
        code_templates = {"python": py_template, **code_templates}
    return py_template, code_templates


def _primary_port_sig(node: dict, direction: str) -> Tuple[str, str]:
    """(type_name, state) for the primary port in ``direction``.

    Prefers the port named by ``node['primary_in']`` / ``node['primary_out']``;
    falls back to the first declared port. Returns ``("any", "any")`` when no
    port exists.
    """
    ports = node.get(direction, {}) or {}
    if not isinstance(ports, dict) or not ports:
        return ("any", "any")

    primary_key = "primary_in" if direction == "inputs" else "primary_out"
    prim_name = node.get(primary_key, "") or ""
    chosen = None
    if prim_name and prim_name in ports:
        chosen = ports[prim_name]
    else:
        chosen = next(iter(ports.values()))

    if not isinstance(chosen, dict):
        return ("any", "any")
    return (
        str(chosen.get("type_name", "any") or "any"),
        str(chosen.get("state", "any") or "any"),
    )


def extract_node_data(node: dict, domain: str) -> tuple:
    py_template, code_templates = _extract_python_template(node)

    node_role = node.get("node_role") or node.get("role") or "function"
    node_type = node.get("node_type") or "function"

    dependencies = node.get("dependencies", []) or []
    if isinstance(dependencies, str):
        dependencies = [dependencies]

    slots = node.get("slots", {}) or {}

    in_type, in_state = _primary_port_sig(node, "inputs")
    out_type, out_state = _primary_port_sig(node, "outputs")

    # configuration_schema carries every extended-state field the loader reads
    # after the initial SELECT — inputs, outputs, slots, postconditions, etc.
    configuration_schema = {
        "inputs": node.get("inputs", {}) or {},
        "outputs": node.get("outputs", {}) or {},
        "slots": slots,
        "bound_slots": node.get("bound_slots", {}) or {},
        "preconditions": node.get("preconditions", []) or [],
        "postconditions": node.get("postconditions", []) or [],
        "effects": node.get("effects", []) or [],
        "type_vars": node.get("type_vars", []) or [],
        "raises": node.get("raises", []) or [],
        "semantic_tags": node.get("semantic_tags", []) or [],
        "topology_type": node.get("topology_type", "") or "",
        "mutation_type": node.get("mutation_type", "") or "",
        "is_context_manager": bool(node.get("is_context_manager", False)),
        "is_public": bool(node.get("is_public", True)),
        "source_provenance": node.get("source_provenance", "") or "",
        "primary_in": node.get("primary_in", "") or "",
        "primary_out": node.get("primary_out", "") or "",
        "code_templates": code_templates,
    }

    return (
        # --- loader contract ---
        node["cell_id"],
        domain,
        node_type,
        node_role,
        node["stage"],
        _json_or_null(node.get("keywords", [])),
        in_type,
        in_state,
        out_type,
        out_state,
        py_template,
        _json_or_null(dependencies),
        _json_or_null(configuration_schema),
        _json_or_null(slots),
        1 if node.get("verified", True) else 0,
        node.get("docstring", "") or "",
        int(node.get("source_priority", 100) or 100),
        # --- extras ---
        domain,
        node.get("doc", "") or "",
        _json_or_null(node.get("inputs", {})),
        _json_or_null(node.get("outputs", {})),
        _json_or_null(code_templates) if code_templates else _json_or_null({"python": py_template}),
        node.get("topology_type", "") or "",
        node.get("mutation_type", "") or "",
        1 if node.get("is_context_manager", False) else 0,
        _json_or_null(node.get("bound_slots", {})),
        _json_or_null(node.get("preconditions", [])),
        _json_or_null(node.get("postconditions", [])),
        _json_or_null(node.get("effects", [])),
        _json_or_null(node.get("type_vars", [])),
        _json_or_null(node.get("raises", [])),
        _json_or_null(node.get("semantic_tags", [])),
        node.get("primary_in", "") or "",
        node.get("primary_out", "") or "",
        1 if node.get("is_public", True) else 0,
        node.get("source_provenance", "") or "",
    )


# --------------------------------------------------------------------------- #
# Tree iteration / schema
# --------------------------------------------------------------------------- #

def _iter_tree_nodes(path: Path) -> Iterable[Tuple[str, dict]]:
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
    files = [p for p in files if p.name not in {"manifest.json", "index.json"}]
    if domain_filter:
        wanted = {d.lower() for d in domain_filter}
        files = [p for p in files if p.stem.lower() in wanted]
    return files


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the ``nodes`` table if missing; add any new columns.

    Note: this does NOT migrate an older ``morphisms`` table. If you compiled
    with a previous version of this script, delete the DB and recompile:
        rm -f trees/lattice.db*
    """
    cur = conn.cursor()
    cur.execute(_CREATE_TABLE_SQL)
    existing = {row[1] for row in cur.execute("PRAGMA table_info(nodes)")}
    for col in _COLUMNS:
        if col not in existing:
            col_type = "INTEGER" if col in _INT_COLUMNS else "TEXT"
            cur.execute(f"ALTER TABLE nodes ADD COLUMN {col} {col_type}")
    conn.commit()


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #

def compile_database(
    output_db: str = DB_PATH, domain_filter: Optional[List[str]] = None
) -> int:
    trees_dir = PROJECT_ROOT / "trees"
    tree_files = _resolve_tree_files(trees_dir, domain_filter)

    if not tree_files:
        print(f"[compile_trees] No tree files found in {trees_dir}"
              + (f" for domains={domain_filter}" if domain_filter else ""))
        return 0

    Path(output_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_db)
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        written = 0
        skipped = 0
        signature_only = 0
        signature_only_by_domain: dict = {}
        warnings: List[str] = []

        for path in tree_files:
            try:
                nodes = list(_iter_tree_nodes(path))
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"[compile_trees] SKIP {path.name}: {exc}", file=sys.stderr)
                continue

            for domain, node in nodes:
                if "cell_id" not in node or "stage" not in node:
                    print(f"[compile_trees] SKIP node in {path.name}: missing cell_id/stage",
                          file=sys.stderr)
                    skipped += 1
                    continue

                row = extract_node_data(node, domain)
                if len(row) != len(_COLUMNS):
                    raise RuntimeError(
                        f"extract_node_data returned {len(row)} fields, "
                        f"schema expects {len(_COLUMNS)}."
                    )

                template = row[10]                      # `code` column
                if not template or not template.strip():
                    signature_only += 1
                    signature_only_by_domain[domain] = (
                        signature_only_by_domain.get(domain, 0) + 1
                    )
                else:
                    aliases = _module_aliases(node.get("dependencies", []) or [])
                    ok, err = _validate_template(
                        template, row[0], node.get("node_role", ""), aliases,
                    )
                    if not ok:
                        warnings.append(f"{path.name}::{row[0]}: {err}")

                cur.execute(INSERT_SQL, row)
                written += 1

        conn.commit()

        print(f"[compile_trees] Compiled {written} node(s) from "
              f"{len(tree_files)} file(s) into {output_db}"
              + (f" ({skipped} skipped)" if skipped else ""))

        if signature_only:
            parts = ", ".join(f"{d}={n}" for d, n in sorted(signature_only_by_domain.items()))
            print(f"[compile_trees] {signature_only} signature-only cell(s) "
                  f"(body synthesized on demand): {parts}")

        if warnings:
            print(f"[compile_trees] {len(warnings)} template warning(s):",
                  file=sys.stderr)
            for w in warnings[:40]:
                print(f"  WARN {w}", file=sys.stderr)
            if len(warnings) > 40:
                print(f"  ... and {len(warnings) - 40} more", file=sys.stderr)

        return written
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="NSTL Tree Compiler")
    parser.add_argument("--output", type=str, default=DB_PATH)
    parser.add_argument("--domains", nargs="*", default=None)
    args = parser.parse_args()
    compile_database(output_db=args.output, domain_filter=args.domains)


if __name__ == "__main__":
    main()
