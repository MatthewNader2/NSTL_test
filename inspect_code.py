#!/usr/bin/env python3
import json

print("=== 1. trees/pillow.json: Filter Cells Dependencies ===")
with open("trees/pillow.json") as f:
    data = json.load(f)
cells = data.get("cells", data) if isinstance(data, dict) else data
cells_list = cells if isinstance(cells, list) else list(cells.values())
for c in cells_list:
    cid = c.get("cell_id", "")
    if any(k in cid for k in ("FIND_EDGES", "CONTOUR", "CUSTOM_KERNEL")):
        print(f"  {cid}:")
        print(f"    dependencies:  {c.get('dependencies')}")
        print(f"    imports:       {c.get('imports')}")
        print(f"    code_template: {c.get('code_template')}")

print("\n=== 2. src/planner.py: _is_terminal_sink_cell & _verify_frontier_step ===")
with open("src/planner.py") as f:
    lines = f.readlines()

def print_function(name, max_lines=25):
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {name}(") or f"def {name}(" in line:
            print(f"--- {name} (starting line {i+1}) ---")
            for j in range(i, min(i + max_lines, len(lines))):
                print(f"{j+1}: {lines[j]}", end="")
                if j > i and lines[j].startswith("    def "):
                    break
            print()
            return

print_function("_is_terminal_sink_cell", 15)
print_function("_verify_frontier_step", 25)
print_function("_is_valid_terminal", 20)

print("\n=== 3. src/synthesis.py: Where dependencies/imports are processed ===")
with open("src/synthesis.py") as f:
    for i, line in enumerate(f):
        if any(w in line for w in ("dependencies", "cell.imports", "derive_import")):
            print(f"  Line {i+1}: {line.strip()}")
