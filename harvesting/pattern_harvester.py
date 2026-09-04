"""
harvesting/pattern_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Loads domain seed patterns directly from consolidated trees/*.json files (source_priority <= 10).
"""

from __future__ import annotations
import glob
import json
import sqlite3
import os
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TREES_DIR = os.path.join(PROJECT_ROOT, "trees")


def load_seeds(trees_dir: Optional[str] = None, domain_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Loads curated seed pattern definitions (source_priority <= 10) from trees/*.json."""
    target_dir = trees_dir or TREES_DIR
    if not os.path.exists(target_dir):
        return []

    seeds: List[Dict[str, Any]] = []
    pattern = os.path.join(target_dir, "*.json")
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath).lower()
        if domain_filter:
            if not any(d.lower() in fname for d in domain_filter):
                continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                cells = data.get("cells", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                for c in cells:
                    if isinstance(c, dict) and c.get("source_priority", 100) <= 10:
                        seeds.append(c)
        except Exception as e:
            print(f"[!] Error loading tree file {fpath}: {e}")

    return seeds


def harvest_core_patterns(db_path: str, domain_filter: Optional[List[str]] = None):
    """Compiles clean, multi-port core patterns dynamically loaded from trees/*.json into SQLite."""
    seeds = load_seeds(domain_filter=domain_filter)
    if not seeds:
        return

    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for p in seeds:
        cid = p["cell_id"]
        domain = p.get("domain_name", "generic")
        role = p.get("node_role", "function")
        stage = p["stage"]
        code = p["code_template"]
        keywords_json = json.dumps(p.get("keywords", p.get("semantic_tags", [])))
        deps_json = json.dumps(p.get("dependencies", []))

        inputs = p.get("inputs", {})
        outputs = p.get("outputs", {})

        first_in = next(iter(inputs.values())) if inputs else {}
        first_out = next(iter(outputs.values())) if outputs else {}

        in_type = first_in.get("type_name", "any")
        in_state = first_in.get("state", "raw")
        out_type = first_out.get("type_name", "None")
        out_state = first_out.get("state", "default")

        cfg = {"inputs": inputs, "outputs": outputs}
        cfg_json = json.dumps(cfg)

        cursor.execute("""
            INSERT OR REPLACE INTO nodes
            (cell_id, domain_name, node_type, node_role, stage, keywords,
             input_type, input_state, output_type, output_state, code,
             dependencies, configuration_schema, verified, source_provenance, source_priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cid, domain, "function", role, stage, keywords_json,
            in_type, in_state, out_type, out_state, code,
            deps_json, cfg_json, 1, "curated_seed", 1
        ))

    conn.commit()
    conn.close()


CORE_CODE_PATTERNS = load_seeds()
