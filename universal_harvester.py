"""
universal_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Universal CLI entrypoint for library introspection and tree generation.
Usage: python3 universal_harvester.py <library_name>
"""
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from universal_harvester import UniversalHarvester

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 universal_harvester.py <library_name>")
        sys.exit(1)

    lib = sys.argv[1]
    harvester = UniversalHarvester(lib)
    out_target = Path("trees") / f"{lib}.json"
    tree = harvester.harvest_and_save(out_target)

    by_archetype = {}
    bridges = []
    for c in tree.cells:
        k = f"Stage {c.stage} | Type: {c.node_type} | Role: {c.node_role}"
        by_archetype[k] = by_archetype.get(k, 0) + 1
        if c.node_role == "tunnel" or c.node_type == "bridge":
            bridges.append(c)

    print(f"\n[NSTL Universal Harvester] Completed harvest of '{lib}':")
    print(f"Total cells generated: {len(tree.cells)}")
    print("\nArchetype breakdown:")
    for k, count in sorted(by_archetype.items()):
        print(f"  {k}: {count} nodes")

    if bridges:
        print(f"\nSample Bridge / Tunnel nodes ({len(bridges)} total):")
        for b in bridges[:5]:
            in_types = [(k, v.type_name, v.state) for k, v in b.inputs.items()]
            out_types = [(k, v.type_name, v.state) for k, v in b.outputs.items()]
            print(f"  {b.cell_id}: {in_types} -> {out_types}")
