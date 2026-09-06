"""
tools/run_universal_harvest.py - Neuro-Symbolic Topological Lattice (NSTL)
Universal Harvester CLI for any Python library.

Usage:
  python -m tools.run_universal_harvest --library cv2 --enrich --compile
  python -m tools.run_universal_harvest --all --enrich --compile
  python -m tools.run_universal_harvest --library <any_package> --output-dir trees/
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.universal_harvester import UniversalHarvester
from src.schema import TreeSchema, CellSchema
from src.template_wiring import repair_wiring_invariant, clean_malformed_template_braces

PRIMARY_DOMAINS: Dict[str, str] = {
    "cv2": "cv2",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "sklearn",
    "matplotlib": "matplotlib.pyplot",
    "python_core": "builtins",
}


def is_cell_wiring_valid(cell_dict: Dict[str, Any]) -> bool:
    template = cell_dict.get("code_template", "")
    placeholders = set(re.findall(r"\{(\w+)\}", template)) - {"output_var"}
    inputs = set(cell_dict.get("inputs", {}).keys())
    return placeholders.issubset(inputs)


def harvest_single_domain(
        domain_name: str,
        package_name: str,
        container_type: Optional[str],
        output_dir: Path,
        enrich: bool = True
) -> Path:
    print(f"\n========================================================")
    print(f"[*] Universally Harvesting Domain: {domain_name} (pkg: {package_name})")
    print(f"========================================================")

    harvester = UniversalHarvester(
        domain_name=domain_name,
        package_name=package_name,
        container_type=container_type
    )

    cells = harvester.harvest_all()
    print(f"[+] Harvested {len(cells)} raw morphism nodes for {domain_name}")

    if enrich:
        enrichment_sources = [
            output_dir / f"{domain_name}.json",
            PROJECT_ROOT / "nstl_enrichment" / "checkpoints" / f"{domain_name}.json"
        ]
        cells = UniversalHarvester.enrich_from_existing_trees(
            domain_name=domain_name,
            new_cells=cells,
            existing_tree_paths=enrichment_sources
        )

    # Validate wiring and syntax
    valid_cells: List[CellSchema] = []
    wiring_issues = 0
    ast_issues = 0

    for c in cells:
        c_dict = c.model_dump()
        if not is_cell_wiring_valid(c_dict):
            repair_wiring_invariant(c_dict, domain_name)
            if not is_cell_wiring_valid(c_dict):
                wiring_issues += 1
                continue
            c = CellSchema(**c_dict)

        # AST syntax check
        dummy_code = re.sub(r'\{[a-zA-Z_][a-zA-Z0-9_]*\}', 'dummy_var', c.code_template)
        try:
            ast.parse(dummy_code)
            valid_cells.append(c)
        except SyntaxError:
            ast_issues += 1

    print(f"[+] Validated {len(valid_cells)} / {len(cells)} cells (wiring dropped: {wiring_issues}, AST dropped: {ast_issues})")

    # Output clean tree JSON
    tree_dict = {
        "domain": domain_name,
        "version": "2.0.0",
        "cells": [c.model_dump() for c in valid_cells]
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{domain_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(tree_dict, f, indent=2)

    print(f"[✓] Saved clean categorized domain tree to: {out_file} ({len(valid_cells)} nodes)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Universal Library Harvester & Morphism Generator")
    parser.add_argument("--library", type=str, help="Name of the library to harvest (e.g. cv2, numpy, pandas)")
    parser.add_argument("--all", action="store_true", help="Harvest all 7 primary NSTL domain libraries")
    parser.add_argument("--package", type=str, help="Underlying Python package name if different from library name")
    parser.add_argument("--container", type=str, help="Carrier container class name override (e.g. Mat, DataFrame)")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "trees"), help="Target output directory")
    parser.add_argument("--enrich", action="store_true", default=True, help="Enrich from existing trees/checkpoints")
    parser.add_argument("--compile", action="store_true", help="Automatically compile into trees/lattice.db")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.all:
        for dom, pkg in PRIMARY_DOMAINS.items():
            harvest_single_domain(
                domain_name=dom,
                package_name=pkg,
                container_type=None,
                output_dir=out_dir,
                enrich=args.enrich
            )
    elif args.library:
        pkg = args.package or PRIMARY_DOMAINS.get(args.library, args.library)
        harvest_single_domain(
            domain_name=args.library,
            package_name=pkg,
            container_type=args.container,
            output_dir=out_dir,
            enrich=args.enrich
        )
    else:
        parser.print_help()
        sys.exit(1)

    if args.compile:
        print("\n[*] Compiling updated domain trees into SQLite database...")
        from tools.compile_trees import compile_database

        compile_database(str(out_dir / "lattice.db"))
        print("[✓] Direct Schema Compilation Complete!")


if __name__ == "__main__":
    main()
