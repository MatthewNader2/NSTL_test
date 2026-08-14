import sys
import os
import unittest

# Ensure src/ is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from unification import UnificationGate, types_unify, TOP_TYPE_SET
from main import infer_goal_output_type

class TestTypePropagationSoundness(unittest.TestCase):
    def test_types_unify_operator(self):
        """Verify formal unification operator unify(tau_expected, tau_actual)."""
        # Exact identity
        self.assertTrue(types_unify("DataFrame", "DataFrame"))
        self.assertTrue(types_unify("Mat", "Mat"))
        
        # Top Type ⊤ Wildcards (any, Any, *)
        self.assertTrue(types_unify("any", "DataFrame"))
        self.assertTrue(types_unify("DataFrame", "any"))
        self.assertTrue(types_unify("*", "int"))
        
        # Incompatible Concrete Types Must NOT Unify (Fail-Closed)
        self.assertFalse(types_unify("DataFrame", "int"))
        self.assertFalse(types_unify("Mat", "dict"))

    def test_synthesis_soundness_rejection(self):
        """
        REGRESSION TEST FOR ROOT CAUSE 3 SOUNDNESS:
        A synthesized mid-chain or goal cell that returns an incompatible type
        MUST BE REJECTED by UnificationGate.validate_synthesis (fail-closed).
        """
        # Bad Cell: returns 'int' when 'DataFrame' is required
        bad_synthesized_cell = {
            "cell_id": "micro_synthesized_bad_cell",
            "type": "micro",
            "stage": 1,
            "inputs": {"type_name": "str", "state": "raw"},
            "outputs": {"type_name": "int", "state": "computed"},
            "domain_implementations": {
                "Python_Core": {
                    "code": "{output_var} = len({input_var})",
                    "dependencies": []
                }
            }
        }

        # Valid Cell: returns 'DataFrame' when 'DataFrame' is required
        good_synthesized_cell = {
            "cell_id": "micro_synthesized_good_cell",
            "type": "micro",
            "stage": 1,
            "inputs": {"type_name": "str", "state": "raw"},
            "outputs": {"type_name": "DataFrame", "state": "computed"},
            "domain_implementations": {
                "Python_Core": {
                    "code": "import pandas as pd\n{output_var} = pd.read_csv({input_var})",
                    "dependencies": []
                }
            }
        }

        # Soundness Verification: bad cell returning 'int' for 'DataFrame' target MUST be REJECTED
        is_valid_bad = UnificationGate.validate_synthesis(
            bad_synthesized_cell,
            expected_inputs="str",
            expected_outputs="DataFrame",
            trees_dir="trees"
        )
        self.assertFalse(is_valid_bad, "Soundness gap! UnificationGate accepted an incompatible output type ('int' instead of 'DataFrame').")

        # Validity Verification: good cell returning 'DataFrame' MUST be ACCEPTED
        is_valid_good = UnificationGate.validate_synthesis(
            good_synthesized_cell,
            expected_inputs="str",
            expected_outputs="DataFrame",
            trees_dir="trees"
        )
        self.assertTrue(is_valid_good, "UnificationGate wrongly rejected a valid matching typestate ('DataFrame').")

    def test_typestate_propagation_inference(self):
        """Verify prompt intent typestate propagation when downstream cell is unconstrained."""
        self.assertEqual(infer_goal_output_type("Read data.csv and drop missing values"), "DataFrame")
        self.assertEqual(infer_goal_output_type("Convert input.jpg to grayscale with opencv"), "Mat")
        self.assertEqual(infer_goal_output_type("Compute dijkstra shortest path algorithm"), "dict")

    def test_lattice_no_any_outputs_invariant(self):
        """Lattice Invariant Test: Harvested Tier 1-4 node files have zero output wildcards."""
        import glob
        import json
        
        harvests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "harvests"))
        enriched_files = glob.glob(os.path.join(harvests_dir, "enriched_*.json")) + glob.glob(os.path.join(harvests_dir, "verified_*.json"))
        
        if not enriched_files:
            self.skipTest("No harvested enriched/verified JSON files found yet.")
            
        any_outputs_count = 0
        for fpath in enriched_files:
            with open(fpath, "r", encoding="utf-8") as f:
                nodes = json.load(f)
            for node in nodes:
                out_type = node.get("outputs", [{}])[0].get("type", "").lower()
                if out_type in ("any", "any_computed"):
                    any_outputs_count += 1
                    
        self.assertEqual(any_outputs_count, 0, f"Lattice invariant violation! Found {any_outputs_count} nodes with 'any' output type.")

    def test_lattice_no_function_name_types_invariant(self):
        """Lattice Invariant Test: No node's input/output type matches its own function name (catches cvtColor bug class)."""
        import glob
        import json
        
        harvests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "harvests"))
        enriched_files = glob.glob(os.path.join(harvests_dir, "enriched_*.json")) + glob.glob(os.path.join(harvests_dir, "verified_*.json"))
        
        if not enriched_files:
            self.skipTest("No harvested enriched/verified JSON files found yet.")
            
        valid_builtin_types = {"dataframe", "series", "ndarray", "mat", "bool", "int", "float", "str", "list", "dict", "set", "tuple", "expanding", "rolling"}
        fn_name_matches = 0
        for fpath in enriched_files:
            with open(fpath, "r", encoding="utf-8") as f:
                nodes = json.load(f)
            for node in nodes:
                cell_id = node.get("cell_id", "")
                fn_name = node.get("name", cell_id.split("_")[-1]).lower()
                
                in_type = node.get("inputs", [{}])[0].get("type", "").lower()
                out_type = node.get("outputs", [{}])[0].get("type", "").lower()
                
                if (in_type == fn_name or out_type == fn_name) and fn_name not in valid_builtin_types:
                    fn_name_matches += 1
                    
        self.assertEqual(fn_name_matches, 0, f"Lattice invariant violation! Found {fn_name_matches} nodes where type_name matches function name.")

if __name__ == "__main__":
    unittest.main()
