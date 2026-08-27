"""
harvesting/pattern_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Dynamically loads domain-agnostic seed patterns and algorithmic primitives from seeds/*.json.
ZERO hardcoded pattern dictionaries in Python code.
"""

from __future__ import annotations
import glob
import json
import sqlite3
import os
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS_DIR = os.path.join(PROJECT_ROOT, "seeds")


def load_seeds(seeds_dir: Optional[str] = None, domain_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Loads seed pattern definitions from external JSON files in seeds/."""
    target_dir = seeds_dir or SEEDS_DIR
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
                if isinstance(data, list):
                    seeds.extend(data)
                elif isinstance(data, dict):
                    seeds.extend(data.get("cells", [data]))
        except Exception as e:
            print(f"[!] Error loading seed file {fpath}: {e}")

    return seeds


def harvest_core_patterns(db_path: str, domain_filter: Optional[List[str]] = None):
    """Compiles clean, multi-port core patterns dynamically loaded from seeds/*.json into SQLite."""
    seeds = load_seeds(domain_filter=domain_filter)
    if not seeds:
        print(f"[!] No seeds found for domain filter: {domain_filter}")
        return

    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for p in seeds:
        cid = p["cell_id"]
        domain = p.get("domain_name", "generic")
        role = p.get("node_role", "function")
        stage = p.get("stage", 2)
        keywords = json.dumps(p.get("keywords", []))

        inputs = p.get("inputs", {})
        outputs = p.get("outputs", {})

        first_in = next(iter(inputs.values())) if inputs else {"type_name": "any", "state": "any"}
        first_out = next(iter(outputs.values())) if outputs else {"type_name": "any", "state": "computed"}

        in_type = first_in.get("type_name", "any")
        in_state = first_in.get("state", "any")
        out_type = first_out.get("type_name", "any")
        out_state = first_out.get("state", "computed")

        code = p.get("code_template", "")
        deps = json.dumps(p.get("dependencies", []))
        schema = json.dumps(inputs)
        verified = p.get("verified", 1)

        cursor.execute("""
            INSERT OR REPLACE INTO nodes (
                cell_id, domain_name, node_type, node_role, stage, keywords,
                input_type, input_state, output_type, output_state,
                code, dependencies, configuration_schema, verified, source_provenance, source_priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (cid, domain, "function", role, stage, keywords, in_type, in_state, out_type, out_state, code, deps, schema, verified, "seeds_json", 4))

    conn.commit()
    conn.close()
    print(f"[+] Successfully harvested {len(seeds)} external seed patterns into {db_path}")


if __name__ == "__main__":
    db = os.path.join(PROJECT_ROOT, "trees", "lattice.db")
    harvest_core_patterns(db)
