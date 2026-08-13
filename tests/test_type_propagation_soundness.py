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

if __name__ == "__main__":
    unittest.main()
