import ast
import unittest
import os
import sys

# Ensure src and root are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from harvesting.pattern_harvester import CORE_CODE_PATTERNS
from unification import UnificationGate, ExecutionContext
from lattice import MicroCell, AlgebraicSignature, PortSignature

FORBIDDEN_CONSTRUCTS = [
    "locals()", "globals()", "hasattr", "getattr",
    "transform_str_or_callable", "normalize_keyword_aggregation"
]

def assert_generated_code_is_clean(code_str: str):
    """Asserts that generated code is clean, idiomatic Python without runtime hacks."""
    for forbidden in FORBIDDEN_CONSTRUCTS:
        assert forbidden not in code_str, f"DUCK-TAPE DETECTED: Found '{forbidden}' in generated code:\n{code_str}"
    
    # Assert AST parses cleanly
    tree = ast.parse(code_str)
    assert len(tree.body) > 0, "Generated empty code"

class TestCodeCleanliness(unittest.TestCase):
    def test_harvester_templates_are_clean(self):
        """Verify all micro-cell templates in pattern_harvester.py contain zero forbidden constructs."""
        for cell_def in CORE_CODE_PATTERNS:
            cell_id = cell_def.get("cell_id", "")
            code = cell_def.get("code", "")
            for forbidden in FORBIDDEN_CONSTRUCTS:
                self.assertNotIn(
                    forbidden, code,
                    f"Forbidden construct '{forbidden}' found in template for cell {cell_id}:\n{code}"
                )

    def test_ml_pipeline_unification_cleanliness(self):
        """Verify unified ML pipeline code is clean and passes static cleanliness checks."""
        context = ExecutionContext(prompt="Load data.csv, train RandomForestClassifier, and save predictions to predictions.csv")

        # 1. Read CSV
        c1 = MicroCell(
            cell_id="PANDAS_READ_CSV_DEFAULT",
            stage=1,
            keywords={"csv", "read"},
            inputs={"input_filename": PortSignature(name="input_filename", signature=AlgebraicSignature("str", "source_identifier"))},
            outputs={"output_var": PortSignature(name="output_var", signature=AlgebraicSignature("DataFrame", "raw"))},
            domain_name="data_processing",
            code_template="import pandas as pd\n{output_var} = pd.read_csv({input_filename})",
            dependencies=["import pandas as pd"]
        )
        code1 = UnificationGate.unify_cell(context, c1)
        assert_generated_code_is_clean(code1)

        # 2. Drop NA
        c2 = MicroCell(
            cell_id="PANDAS_DROPNA_DEFAULT",
            stage=2,
            keywords={"dropna"},
            inputs={"input_var": PortSignature(name="input_var", signature=AlgebraicSignature("DataFrame", "raw"))},
            outputs={"output_var": PortSignature(name="output_var", signature=AlgebraicSignature("DataFrame", "cleaned"))},
            domain_name="data_processing",
            code_template="{output_var} = {input_var}.dropna()",
            dependencies=["import pandas as pd"]
        )
        code2 = UnificationGate.unify_cell(context, c2)
        assert_generated_code_is_clean(code2)

        # 3. Fit Model
        c3 = MicroCell(
            cell_id="SKLEARN_RANDOM_FOREST_FIT",
            stage=2,
            keywords={"fit", "randomforest"},
            inputs={"input_var": PortSignature(name="input_var", signature=AlgebraicSignature("DataFrame", "cleaned"))},
            outputs={"output_var": PortSignature(name="output_var", signature=AlgebraicSignature("Model", "trained"))},
            domain_name="machine_learning",
            code_template=(
                "from sklearn.ensemble import RandomForestClassifier\n\n"
                "X_{output_var} = {input_var}.iloc[:, :-1]\n"
                "y_{output_var} = {input_var}.iloc[:, -1]\n"
                "model = RandomForestClassifier()\n"
                "model.fit(X_{output_var}, y_{output_var})\n"
                "{output_var} = model"
            ),
            dependencies=["import sklearn"]
        )
        code3 = UnificationGate.unify_cell(context, c3)
        assert_generated_code_is_clean(code3)

        # 4. Predict
        c4 = MicroCell(
            cell_id="SKLEARN_MODEL_PREDICT",
            stage=2,
            keywords={"predict"},
            inputs={"input_var": PortSignature(name="input_var", signature=AlgebraicSignature("Model", "trained"))},
            outputs={"output_var": PortSignature(name="output_var", signature=AlgebraicSignature("ndarray", "predictions"))},
            domain_name="machine_learning",
            code_template="{output_var} = {input_var}.predict({input_var})",
            dependencies=["import sklearn"]
        )
        code4 = UnificationGate.unify_cell(context, c4)
        assert_generated_code_is_clean(code4)

        # 5. Score
        c5 = MicroCell(
            cell_id="SKLEARN_SCORE_DEFAULT",
            stage=2,
            keywords={"score"},
            inputs={"input_var": PortSignature(name="input_var", signature=AlgebraicSignature("Model", "trained"))},
            outputs={"output_var": PortSignature(name="output_var", signature=AlgebraicSignature("float", "accuracy"))},
            domain_name="machine_learning",
            code_template="{output_var} = {input_var}.score({input_var}, {input_var})",
            dependencies=["import sklearn"]
        )
        code5 = UnificationGate.unify_cell(context, c5)
        assert_generated_code_is_clean(code5)

        # 6. Save Predictions
        c6 = MicroCell(
            cell_id="PANDAS_SAVE_PREDICTIONS",
            stage=3,
            keywords={"save", "to_csv"},
            inputs={"output_filename": PortSignature(name="output_filename", signature=AlgebraicSignature("str", "dest_identifier"))},
            outputs={"output_var": PortSignature(name="output_var", signature=AlgebraicSignature("str", "filepath_written"))},
            domain_name="machine_learning",
            code_template=(
                "import pandas as pd\n\n"
                "_out_df = pd.DataFrame({'prediction': {input_var}})\n"
                "{output_var} = _out_df.to_csv({output_filename}, index=False)"
            ),
            dependencies=["import pandas as pd"]
        )
        code6 = UnificationGate.unify_cell(context, c6)
        assert_generated_code_is_clean(code6)

        full_program = "\n".join([code1, code2, code3, code4, code5, code6])
        assert_generated_code_is_clean(full_program)

if __name__ == "__main__":
    unittest.main()
