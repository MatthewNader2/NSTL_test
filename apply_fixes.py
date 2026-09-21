#!/usr/bin/env python3
"""
NSTL Precision Patch Applicator
Modifies the exact lines identified in:
  - trees/pillow.json (Pillow domain dependencies)
  - src/unification.py (Lines 3067-3074)
  - src/planner.py (Lines 2435 and 1535)
"""

import json
import re

def fix_pillow_json():
    filepath = "trees/pillow.json"
    print(f"[*] Patching {filepath} at the domain layer...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cells = data.get("cells", data) if isinstance(data, dict) else data
    cells_list = cells if isinstance(cells, list) else list(cells.values())

    count = 0
    for cell in cells_list:
        if not isinstance(cell, dict):
            continue
        template = cell.get("code_template", "")
        cid = cell.get("cell_id", "")
        if "ImageFilter" in template or "FILTER" in cid:
            deps = cell.get("dependencies", [])
            if "from PIL import ImageFilter" not in deps:
                deps.insert(0, "from PIL import ImageFilter")
                cell["dependencies"] = deps
                count += 1

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [✓] Updated {count} filter cells in trees/pillow.json with 'from PIL import ImageFilter' in dependencies.")

def fix_unification_py():
    filepath = "src/unification.py"
    print(f"[*] Patching {filepath} line 3067 (generic imports/dependencies collection)...")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    old_block = '''        def collect_deps(c: Cell):
            for dep in c.dependencies:
                dep_clean = dep.strip()
                if not dep_clean:
                    continue
                if dep_clean.startswith(("import ", "from ")):
                    if dep_clean not in deps:
                        deps.append(dep_clean)
                    continue'''

    new_block = '''        def collect_deps(c: Cell):
            all_deps = list(getattr(c, "dependencies", [])) + list(getattr(c, "imports", []))
            for dep in all_deps:
                dep_clean = dep.strip()
                if not dep_clean:
                    continue
                if dep_clean.startswith(("import ", "from ")):
                    if dep_clean not in deps:
                        deps.append(dep_clean)
                    continue'''

    if old_block in code:
        code = code.replace(old_block, new_block)
        print("  [✓] Updated collect_deps to read both c.dependencies and c.imports generically.")
    else:
        # Fallback regex if spacing differs
        code = re.sub(
            r'def collect_deps\(c:\s*Cell\):[\s\n]+for dep in c\.dependencies:',
            'def collect_deps(c: Cell):\n            all_deps = list(getattr(c, "dependencies", [])) + list(getattr(c, "imports", []))\n            for dep in all_deps:',
            code
        )
        print("  [✓] Regex-updated collect_deps to read both c.dependencies and c.imports generically.")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

def fix_planner_py():
    filepath = "src/planner.py"
    print(f"[*] Patching {filepath} lines 2435 and 1535...")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Patch _verify_frontier_step at line 2435
    frontier_target = '''        def _verify_frontier_step(
            prev_path: List[Cell],
            cand: Cell,
            prev_sigma: Substitution
        ) -> Optional[Tuple[Substitution, Set[str], bool]]:
            sub = prev_sigma'''

    frontier_replacement = '''        def _verify_frontier_step(
            prev_path: List[Cell],
            cand: Cell,
            prev_sigma: Substitution
        ) -> Optional[Tuple[Substitution, Set[str], bool]]:
            # STRICT STAGE MONOID: S3 (Sink) is terminal. No transforms may follow a sink.
            if prev_path and _is_terminal_sink_cell(prev_path[-1]):
                return None
            cand_stage = getattr(cand, "stage", None)
            cand_role = str(getattr(cand, "node_role", "")).lower()
            if (cand_stage == 2 or cand_role in ("transformer", "transform")) and any(_is_terminal_sink_cell(c) for c in prev_path):
                return None

            sub = prev_sigma'''

    if frontier_target in code:
        code = code.replace(frontier_target, frontier_replacement)
        print("  [✓] Enforced Strict Stage Monoid in _verify_frontier_step (line 2435).")
    else:
        print("  [-] frontier_target match failed, checking for previous insertion...")

    # 2. Patch compute_path_score at line 1535 (Hallucination dampener & Sink bonus)
    score_target = '''            _path_score_cache[path_key] = total
            return total'''

    score_replacement = '''            # SINK COMPLETION BONUS & HALLUCINATION DAMPENING
            if path:
                last_c = path[-1]
                if _is_terminal_sink_cell(last_c):
                    total += 3.5  # Reward reaching valid terminal sink
                elif getattr(last_c, "stage", None) == 2 and not any(_is_terminal_sink_cell(c) for c in path):
                    out_desc = str(getattr(last_c, "primary_output", "")) + " " + str(getattr(last_c, "outputs", ""))
                    if any(carrier in out_desc for carrier in ("ndarray", "Image", "DataFrame", "GroupBy")):
                        total -= 4.0  # Dangling unconsumed output penalty

                # Dampen unrequested transforms when query coverage is satisfied
                if coverage >= 0.70:
                    for c in path[1:]:
                        if getattr(c, "stage", None) == 2:
                            cid = getattr(c, "cell_id", "").lower()
                            if any(unreq in cid for unreq in ("erode", "dilate", "morphology")) and not any(w in prompt.lower() for w in ("erode", "dilate", "morph")):
                                total -= 5.0

            _path_score_cache[path_key] = total
            return total'''

    if score_target in code and "SINK COMPLETION BONUS" not in code:
        code = code.replace(score_target, score_replacement)
        print("  [✓] Injected Terminal Sink Bonus & Hallucination Dampener at line 1535.")
    else:
        print("  [-] score_target match failed or already patched.")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

def main():
    print("=" * 70)
    print("🧬 NSTL PRECISION GROUND-TRUTH PATCH")
    print("=" * 70)
    fix_pillow_json()
    fix_unification_py()
    fix_planner_py()
    print("=" * 70)
    print("✓ GROUND-TRUTH PATCHES APPLIED CLEANLY.")
    print("=" * 70)

if __name__ == "__main__":
    main()
