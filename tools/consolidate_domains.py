"""
tools/consolidate_domains.py
Consolidates all scattered seeds, trees, and harvests into exactly ONE
TreeSchema JSON file per domain under trees/{domain}.json.
"""

import json
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import CellSchema, TreeSchema, PortSchema
from tools.compile_trees import _iter_cells_to_compile, _validate_template, sanitize_type_name, determine_stage

CONTAINER_TYPES = {
    "pandas": "DataFrame",
    "cv2": "Mat",
    "sklearn": "ndarray",
    "matplotlib": "Figure",
    "numpy": "ndarray",
    "scipy": "ndarray",
    "python_core": "any"
}

DOMAIN_CONFIGS = [
    {
        "domain": "pandas",
        "seed_file": "seeds/pandas_seeds.json",
        "tree_files": ["trees/pandas_tree.json"],
        "harvest_files": ["harvests/verified_pandas.json", "harvests/enriched_pandas.json", "harvests/structural_pandas.json"],
        "default_deps": ["import pandas as pd"]
    },
    {
        "domain": "cv2",
        "seed_file": "seeds/cv2_seeds.json",
        "tree_files": ["trees/cv2_tree.json"],
        "harvest_files": ["harvests/verified_cv2.json", "harvests/enriched_cv2.json", "harvests/structural_cv2.json"],
        "default_deps": ["import cv2"]
    },
    {
        "domain": "sklearn",
        "seed_file": "seeds/sklearn_seeds.json",
        "tree_files": ["trees/sklearn_tree.json"],
        "harvest_files": ["harvests/verified_sklearn.json", "harvests/enriched_sklearn.json", "harvests/structural_sklearn.json"],
        "default_deps": ["import sklearn"]
    },
    {
        "domain": "matplotlib",
        "seed_file": "seeds/matplotlib_seeds.json",
        "tree_files": ["trees/matplotlib_tree.json"],
        "harvest_files": [],
        "default_deps": ["import matplotlib.pyplot as plt"]
    },
    {
        "domain": "numpy",
        "seed_file": None,
        "tree_files": ["trees/numpy_tree.json"],
        "harvest_files": ["harvests/verified_numpy.json", "harvests/enriched_numpy.json", "harvests/structural_numpy.json"],
        "default_deps": ["import numpy as np"]
    },
    {
        "domain": "scipy",
        "seed_file": None,
        "tree_files": ["trees/scipy_tree.json"],
        "harvest_files": ["harvests/verified_scipy.json", "harvests/enriched_scipy.json", "harvests/structural_scipy.json"],
        "default_deps": ["import scipy"]
    },
    {
        "domain": "python_core",
        "seed_file": "seeds/python_core_seeds.json",
        "tree_files": ["trees/builtins_tree.json", "trees/macro_tree.json"],
        "harvest_files": [],
        "default_deps": []
    }
]

def normalize_ports(raw_ports: Any, default_type: str = "any", default_state: str = "raw") -> Dict[str, PortSchema]:
    ports = {}
    if isinstance(raw_ports, dict):
        for k, v in raw_ports.items():
            if isinstance(v, dict):
                t = sanitize_type_name(v.get("type_name", v.get("type", default_type)))
                s = v.get("state", default_state)
                req = v.get("required", True)
                def_v = v.get("default_value", v.get("default"))
                ports[k] = PortSchema(
                    type_name=t,
                    state=str(s),
                    required=bool(req),
                    default_value=str(def_v) if def_v is not None else None,
                    description=v.get("description", v.get("doc", ""))
                )
            elif isinstance(v, str):
                ports[k] = PortSchema(type_name=sanitize_type_name(v), state=default_state)
    elif isinstance(raw_ports, list):
        for item in raw_ports:
            if isinstance(item, dict) and "name" in item:
                name = item["name"]
                t = sanitize_type_name(item.get("type_name", item.get("type", default_type)))
                s = item.get("state", default_state)
                req = item.get("required", True)
                def_v = item.get("default_value", item.get("default"))
                ports[name] = PortSchema(
                    type_name=t,
                    state=str(s),
                    required=bool(req),
                    default_value=str(def_v) if def_v is not None else None,
                    description=item.get("description", item.get("doc", item.get("param_doc", "")))
                )
    return ports

def consolidate_domain(config: Dict[str, Any], output_dir: Path) -> TreeSchema:
    domain = config["domain"]
    default_container = CONTAINER_TYPES.get(domain, "any")
    cells_map: Dict[str, CellSchema] = {}

    # 1. Load curated seeds (highest priority: 1)
    seed_file = config.get("seed_file")
    if seed_file and os.path.exists(os.path.join(PROJECT_ROOT, seed_file)):
        with open(os.path.join(PROJECT_ROOT, seed_file), "r", encoding="utf-8") as f:
            seeds = json.load(f)
            for s in seeds:
                cid = s["cell_id"].strip().upper()
                c = CellSchema(
                    cell_id=cid,
                    stage=s["stage"],
                    inputs=normalize_ports(s.get("inputs", {}), default_type=default_container, default_state="any"),
                    outputs=normalize_ports(s.get("outputs", {}), default_type=default_container, default_state="raw"),
                    code_template=s["code_template"],
                    dependencies=s.get("dependencies", config["default_deps"]),
                    semantic_tags=s.get("semantic_tags", []),
                    keywords=s.get("keywords", []),
                    docstring=s.get("docstring", ""),
                    domain_name=domain,
                    node_type=s.get("node_type", "function"),
                    node_role=s.get("node_role", "function"),
                    source_priority=1
                )
                cells_map[cid] = c

    # 2. Ingest tree files and harvests
    source_files = config["tree_files"] + config["harvest_files"]
    for rel_path in source_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Warning reading {rel_path}: {e}")
            continue

        nodes = data.get("nodes", data.get("cells", data)) if isinstance(data, dict) else data
        if not isinstance(nodes, list):
            continue

        for raw_node in nodes:
            if not isinstance(raw_node, dict):
                continue

            for raw_cell in _iter_cells_to_compile(raw_node):
                cid = raw_cell.get("cell_id", "").upper().strip()
                if not cid or cid.startswith("_"):
                    continue
                if cid in cells_map and cells_map[cid].source_priority == 1:
                    continue  # Protect curated seeds

                cid_lower = cid.lower()
                noise = ("arithmeticerror", "baseexception", "add_note", "lookuperror",
                         "exception", "warning", "error", "zombie")
                if any(p in cid_lower for p in noise):
                    continue

                code = raw_cell.get("code_template") or raw_cell.get("code", "")
                if not code:
                    continue

                node_role = raw_cell.get("node_role", "macro" if raw_cell.get("node_type") == "macro" else "function")
                is_valid, _ = _validate_template(code, cid, node_role=node_role)
                if not is_valid:
                    continue

                in_ports = normalize_ports(raw_cell.get("inputs", {}), default_type=default_container, default_state="any")
                out_ports = normalize_ports(raw_cell.get("outputs", {}), default_type=default_container, default_state="processed")

                first_in_type = next(iter(in_ports.values())).type_name if in_ports else default_container
                first_out_type = next(iter(out_ports.values())).type_name if out_ports else default_container
                stage = determine_stage(cid, code, first_in_type, first_out_type, raw_cell.get("stage"))

                if stage == 1:
                    if not in_ports:
                        in_ports = {"filepath": PortSchema(type_name="str", state="source_identifier")}
                    else:
                        first_p = next(iter(in_ports.values()))
                        if first_p.type_name in ("str", "any"):
                            first_p.state = "source_identifier"
                elif stage == 3:
                    if any(k in cid.lower() for k in ("to_csv", "imwrite", "savefig", "to_parquet", "to_json")):
                        if out_ports:
                            next(iter(out_ports.values())).state = "filepath_written"

                if not in_ports:
                    in_ports = {"data": PortSchema(type_name=default_container, state="any")}
                if not out_ports:
                    out_ports = {"output_data": PortSchema(type_name=default_container, state="processed")}

                kws = list(set(raw_cell.get("keywords", []) + [w for w in re.split(r"[_ ]", cid.lower()) if len(w) > 2]))
                deps = raw_cell.get("dependencies", config["default_deps"])

                try:
                    c = CellSchema(
                        cell_id=cid,
                        stage=stage,
                        inputs=in_ports,
                        outputs=out_ports,
                        code_template=code,
                        dependencies=deps,
                        semantic_tags=kws,
                        keywords=kws,
                        docstring=raw_cell.get("docstring", raw_cell.get("description", "")),
                        domain_name=domain,
                        node_type=raw_cell.get("node_type", "function"),
                        node_role=node_role,
                        source_priority=100
                    )
                    cells_map[cid] = c
                except Exception:
                    pass

    tree_schema = TreeSchema(
        domain=domain,
        version="1.0.0",
        cells=list(cells_map.values())
    )

    out_file = output_dir / f"{domain}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(tree_schema.model_dump_json(indent=2))

    print(f"[+] Consolidated domain '{domain}' -> {out_file} ({len(tree_schema.cells)} cells, {sum(1 for c in tree_schema.cells if c.source_priority == 1)} seeds)")
    return tree_schema

def main():
    trees_dir = PROJECT_ROOT / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(" NSTL DOMAIN CONSOLIDATION — SINGLE JSON FILE PER DOMAIN")
    print("======================================================================")

    for cfg in DOMAIN_CONFIGS:
        consolidate_domain(cfg, trees_dir)

    print("\n[*] Consolidation complete for all domains!")

if __name__ == "__main__":
    main()
