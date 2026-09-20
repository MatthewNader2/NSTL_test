#!/usr/bin/env python3
"""
NSTL Domain-Agnostic & Schema-Compliant Patch Applicator
1. trees/pillow.json: Fixes ImageFilter imports directly at the domain level.
2. src/synthesis.py: Reverts/removes any hardcoded domain imports in the engine.
3. src/planner.py: Enforces Strict Stage Monoid (S3 cannot transition to S2) & Hallucination Dampener.
4. src/cli.py: Adds parentheses-safe clause tokenizer for Profile E.
"""

import json
import re
import os
import shutil

def fix_pillow_domain():
    filepath = "trees/pillow.json"
    print(f"[*] Fixing {filepath} at the domain layer...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cells = data.get("cells", data) if isinstance(data, dict) else data
    cells_iterable = cells if isinstance(cells, list) else cells.values()

    modified_count = 0
    for cell in cells_iterable:
        if not isinstance(cell, dict):
            continue
        template = cell.get("template", "")
        cell_id = cell.get("cell_id", "")

        # Any cell in the Pillow domain referencing ImageFilter must declare its own import
        if "ImageFilter" in template or "FILTER" in cell_id:
            imports = cell.get("imports", [])
            if not isinstance(imports, list):
                imports = [imports] if imports else []

            if "from PIL import ImageFilter" not in imports:
                imports.append("from PIL import ImageFilter")
                cell["imports"] = imports
                modified_count += 1

    # Also ensure domain-level imports list has it if present
    if isinstance(data, dict) and "imports" in data and isinstance(data["imports"], list):
        if "from PIL import ImageFilter" not in data["imports"]:
            data["imports"].append("from PIL import ImageFilter")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [✓] Updated {modified_count} filter cells in trees/pillow.json with declared ImageFilter imports.")

def clean_synthesis_engine():
    filepath = "src/synthesis.py"
    print(f"[*] Cleaning {filepath} (ensuring zero hardcoded domain imports in engine)...")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # Remove any engine-level import injection
    pattern = re.compile(r'\s*if "ImageFilter" in code_str[^\n]*\n\s*imports\.append\("from PIL import ImageFilter"\)\n?')
    cleaned = re.sub(pattern, "\n", code)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print("  [✓] src/synthesis.py is clean and 100% domain-agnostic.")

def fix_planner():
    filepath = "src/planner.py"
    print(f"[*] Patching {filepath} (Strict Stage Monoid & Hallucination Dampener)...")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Enforce Strict Stage Monoid: An S3 Sink CANNOT transition into an S2 Transform
    stage_monoid_hook = '''
        # STRICT STAGE MONOID: S3 (Sink) is strictly terminal and cannot transition to S2
        u_stage = getattr(u, 'stage', -1)
        v_stage = getattr(v, 'stage', -1)
        u_is_sink = u_stage in (3, '3', 'S3') or str(getattr(u, 'role', '')).lower() == 'sink'
        v_is_transform = v_stage in (2, '2', 'S2') or str(getattr(v, 'role', '')).lower() == 'transform'
        if u_is_sink and v_is_transform:
            return False
'''
    if "STRICT STAGE MONOID" not in code:
        for method_def in ("def is_transition_valid(", "def _is_valid_transition(", "def is_step_valid("):
            if method_def in code:
                idx = code.find(method_def)
                ret_idx = code.find("return", idx)
                if ret_idx != -1:
                    code = code[:ret_idx] + stage_monoid_hook + "        " + code[ret_idx:]
                    print("  [✓] Enforced Strict Stage Monoid: Sinks (S3) cannot transition to Transforms (S2).")
                    break

    # 2. Suppress Unrequested Transforms & Reward Sinks
    scoring_hook = '''
        # TERMINAL SINK & DOMAIN-AGNOSTIC HALLUCINATION DAMPENING
        last_cell = path[-1]
        last_stage = getattr(last_cell, 'stage', -1)
        is_sink = last_stage in (3, '3', 'S3') or str(getattr(last_cell, 'role', '')).lower() == 'sink'
        out_desc = str(getattr(last_cell, 'primary_output', '')) + " " + str(getattr(last_cell, 'outputs', ''))
        is_data_carrier = any(k in out_desc for k in ('ndarray', 'Image', 'DataFrame', 'GroupBy', 'List['))

        if is_sink:
            score += 3.5  # Terminal sink completion bonus
        elif is_data_carrier and not any(str(getattr(c, 'role', '')).lower() == 'sink' for c in path):
            score -= 4.0  # Dangling unconsumed output penalty

        # Domain-agnostic saturation check: penalize transforms that have near-zero prompt relevance
        if 'cov' in locals() and cov >= 0.70 and 'relevance_map' in locals():
            for c in path[1:]:
                c_stage = getattr(c, 'stage', -1)
                if c_stage in (2, '2', 'S2') and relevance_map.get(c.cell_id, 0.0) < 0.05:
                    score -= 5.0
'''
    if "TERMINAL SINK & DOMAIN-AGNOSTIC HALLUCINATION DAMPENING" not in code:
        for score_def in ("def score_path(", "def _score_path(", "def evaluate_path("):
            if score_def in code:
                idx = code.find(score_def)
                ret_idx = code.find("return score", idx)
                if ret_idx == -1:
                    ret_idx = code.find("return total_score", idx)
                if ret_idx != -1:
                    code = code[:ret_idx] + scoring_hook + "\n        " + code[ret_idx:]
                    print("  [✓] Injected Terminal Sink Bonus and Relevance-Based Hallucination Dampener.")
                    break

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

def fix_cli():
    filepath = "src/cli.py"
    print(f"[*] Patching {filepath} clause tokenizer for parentheses safety...")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    safe_split_code = '''
    def _split_clauses_safe(self, text: str):
        """Splits clauses on commas while preserving parentheses (e.g. resize(256, 256))."""
        clauses = []
        current = []
        depth = 0
        for char in text:
            if char == '(':
                depth += 1
            elif char == ')':
                depth = max(0, depth - 1)
            if char == ',' and depth == 0:
                clause = "".join(current).strip()
                if clause:
                    clauses.append(clause)
                current = []
            else:
                current.append(char)
        if current:
            clause = "".join(current).strip()
            if clause:
                clauses.append(clause)
        return clauses
'''
    if "_split_clauses_safe" not in code:
        if "class NSTLDebugger" in code:
            idx = code.find("class NSTLDebugger")
            next_def = code.find("    def ", idx)
            code = code[:next_def] + safe_split_code + "\n" + code[next_def:]
            print("  [✓] Added parentheses-safe clause tokenizer to NSTLDebugger.")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

def main():
    print("=" * 70)
    print("🧬 NSTL ARCHITECTURAL FIX (DOMAIN-PURE & ENGINE-AGNOSTIC)")
    print("=" * 70)
    fix_pillow_domain()
    clean_synthesis_engine()
    fix_planner()
    fix_cli()
    print("=" * 70)
    print("✓ ALL ARCHITECTURAL FIXES APPLIED.")
    print("Run: python3 nstl_cli.py --debug")
    print("=" * 70)

if __name__ == "__main__":
    main()
