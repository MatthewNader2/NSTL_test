import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from lattice import LatticeOrchestrator, MacroCell, MicroCell


def test():
    trees_dir = os.path.join(ROOT_DIR, "trees")
    orchestrator = LatticeOrchestrator(trees_directory=trees_dir, active_domain="Python_Core")
    
    assert len(orchestrator.loaded_cells) > 0, "No cells loaded from database"
    assert "PANDAS_READ_CSV" in orchestrator.loaded_cells, "PANDAS_READ_CSV not found in lattice"
    
    read_cell = orchestrator.loaded_cells["PANDAS_READ_CSV"]
    print("Read cell primary input:", read_cell.primary_input)
    print("Read cell primary output:", read_cell.primary_output)
    
    assert read_cell.primary_input.type_name == "str"
    assert read_cell.primary_output.type_name == "DataFrame"
    print("All lattice tests passed!")


if __name__ == "__main__":
    test()
