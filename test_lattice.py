import os
import json
from lattice import LatticeOrchestrator, MacroCell, MicroCell

def test():
    # 1. Create a dummy trees directory with a dummy json
    os.makedirs("test_trees", exist_ok=True)
    with open("test_trees/dummy.json", "w") as f:
        json.dump({
            "domain_name": "Python_Core",
            "cells": [
                {
                    "cell_id": "micro_test",
                    "type": "micro",
                    "domain_implementations": {
                        "Python_Core": {
                            "code": "print('hello python')"
                        },
                        "Rust_Standard": {
                            "code": "println!('hello rust')"
                        }
                    }
                },
                {
                    "cell_id": "macro_test",
                    "type": "macro",
                    "algorithmic_steps": ["1. Do something"]
                }
            ]
        }, f)

    # 2. Instantiate orchestrator
    orchestrator = LatticeOrchestrator(trees_directory="test_trees", active_domain="Python_Core")
    
    # 3. Verify
    assert "micro_test" in orchestrator.loaded_cells
    assert "macro_test" in orchestrator.loaded_cells
    
    micro = orchestrator.loaded_cells["micro_test"]
    macro = orchestrator.loaded_cells["macro_test"]
    
    print("Micro code:", micro.code_template)
    print("Macro algorithmic_steps:", macro.algorithmic_steps)
    
    assert micro.code_template == "print('hello python')"
    assert len(macro.algorithmic_steps) == 1
    assert macro.algorithmic_steps[0] == "1. Do something"
    print("All tests passed!")

if __name__ == "__main__":
    test()
