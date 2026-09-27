"""
tests/test_phase7_gevr_verification.py - Neuro-Symbolic Topological Lattice (NSTL)
Phase 7 GEVR Sandbox Task Verification Test Suite:
  1. Phase-1 Cell Postconditions: ndim and typestate checks (PostconditionVerificationError on violation).
  2. Phase-1 Cell Postconditions: DataFrame has_nans and is_deduped validation.
  3. Terminal Intent Verification: Model .fit() on split training partition succeeds.
  4. Terminal Intent Verification: Unfitted model raises PostconditionVerificationError.
  5. Terminal Intent Verification: Model fitted on unsplit data violates split training intent.
  6. Terminal Intent Verification: Image egress with rendered annotations succeeds.
  7. Terminal Intent Verification: Image egress saving unannotated raw input raises PostconditionVerificationError.
  8. GEVR Self-Repair Cycle: Diagnostic PostconditionVerificationError signal drives automated repair.
  9. Tabular and Visualization Egress Intent Verification.
  10. End-to-End UnificationGate Synthesis with Auto-Generated VerificationContract.
"""

import ast
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.errors import PostconditionVerificationError, DataflowExecutionError, ArtifactMaterializationError
from src.gevr_sandbox import GEVRSandbox, verify_postconditions
from src.unification import UnificationGate, VerificationContract, ExecutionContext
from src.lattice import LatticeOrchestrator

DB_PATH = str(PROJECT_ROOT / "trees" / "lattice.db")


class TestPhase7GEVRVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = LatticeOrchestrator()
        cls.orchestrator.load_from_database(DB_PATH)
        cls.orchestrator.build_topology()
        cls.sandbox = GEVRSandbox(timeout_seconds=10)
        cls.temp_dir = tempfile.mkdtemp(prefix="nstl_phase7_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_phase1_postcondition_ndim_and_state(self):
        """
        Verify that Phase-1 postconditions (ndim == 2, state == gray, state == color_bgr)
        are enforced against execution globals and raise PostconditionVerificationError on violation.
        """
        # A. Passing ndim check
        spec_ndim_pass = {
            "cell_checks": [
                {"cell_id": "OPENCV.CVTCOLOR", "target_var": "gray_img", "property": "ndim", "operator": "==", "value": 2}
            ]
        }
        exec_scope = {"gray_img": np.zeros((100, 100), dtype=np.uint8)}
        verify_postconditions(exec_scope, spec_ndim_pass)

        # B. Failing ndim check (array has ndim 3 instead of 2)
        exec_scope_fail = {"gray_img": np.zeros((100, 100, 3), dtype=np.uint8)}
        with self.assertRaises(PostconditionVerificationError) as ctx:
            verify_postconditions(exec_scope_fail, spec_ndim_pass)
        self.assertIn("ndim == 2", str(ctx.exception))
        self.assertIn("'gray_img.ndim' is 3", str(ctx.exception))

        # C. Passing and failing state == gray check
        spec_state_gray = {
            "cell_checks": [
                {"cell_id": "CV2_COLOR_GRAY", "target_var": "var_2", "property": "state", "value": "gray"}
            ]
        }
        verify_postconditions({"var_2": np.zeros((50, 50), dtype=np.uint8)}, spec_state_gray)
        with self.assertRaises(PostconditionVerificationError) as ctx_gray:
            verify_postconditions({"var_2": np.zeros((50, 50, 3), dtype=np.uint8)}, spec_state_gray)
        self.assertIn("state == gray", str(ctx_gray.exception))

        # D. Passing and failing state == color_bgr check
        spec_state_color = {
            "cell_checks": [
                {"cell_id": "CV2_IMREAD", "target_var": "var_1", "property": "state", "value": "color_bgr"}
            ]
        }
        verify_postconditions({"var_1": np.zeros((50, 50, 3), dtype=np.uint8)}, spec_state_color)
        with self.assertRaises(PostconditionVerificationError) as ctx_color:
            verify_postconditions({"var_1": np.zeros((50, 50), dtype=np.uint8)}, spec_state_color)
        self.assertIn("state == color_bgr", str(ctx_color.exception))

    def test_02_phase1_postcondition_pandas_dropna(self):
        """
        Verify that has_nans == False and is_deduped == True postconditions validate DataFrames.
        """
        spec = {
            "cell_checks": [
                {"cell_id": "PANDAS.DATAFRAME.DROPNA", "target_var": "df_clean", "property": "has_nans", "value": False},
                {"cell_id": "PANDAS.DATAFRAME.DROP_DUPLICATES", "target_var": "df_clean", "property": "is_deduped", "value": True}
            ]
        }

        # Passing clean DataFrame
        clean_df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        verify_postconditions({"df_clean": clean_df}, spec)

        # Failing DataFrame with NaNs
        nan_df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, np.nan, 6.0]})
        with self.assertRaises(PostconditionVerificationError) as ctx_nan:
            verify_postconditions({"df_clean": nan_df}, spec)
        self.assertIn("has_nans == False", str(ctx_nan.exception))
        self.assertIn("contains NaNs", str(ctx_nan.exception))

        # Failing DataFrame with duplicates
        dup_df = pd.DataFrame({"a": [1, 1, 3], "b": [4.0, 4.0, 6.0]})
        with self.assertRaises(PostconditionVerificationError) as ctx_dup:
            verify_postconditions({"df_clean": dup_df}, spec)
        self.assertIn("is_deduped == True", str(ctx_dup.exception))
        self.assertIn("contains duplicates", str(ctx_dup.exception))

    def test_03_terminal_model_fitted_success(self):
        """
        Verify that an estimator fitted on the split training partition passes terminal verification.
        """
        code = """
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
var_final = clf
"""
        contract = VerificationContract(
            terminal_checks=[{
                "type": "model_fit_split",
                "cell_id": "SKLEARN.ENSEMBLE.RANDOMFORESTCLASSIFIER.FIT",
                "model_var": "clf",
                "feature_var": "X_train",
                "target_var": "y_train",
                "expected_train_feature_var": "X_train",
                "unsplit_feature_var": "X",
            }]
        )

        res = self.sandbox.execute(code, verification_spec=contract)
        self.assertTrue(res["success"], f"Execution failed: {res.get('error')}")
        self.assertEqual(res.get("error"), "")

    def test_04_terminal_model_unfit_failure_signal(self):
        """
        Verify that an instantiated estimator that was never .fit() raises PostconditionVerificationError.
        """
        code = """
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, random_state=42)
var_final = clf
"""
        contract = VerificationContract(
            terminal_checks=[{
                "type": "model_fit_split",
                "cell_id": "SKLEARN.ENSEMBLE.RANDOMFORESTCLASSIFIER.FIT",
                "model_var": "clf",
                "feature_var": "X_train",
                "target_var": "y_train",
            }]
        )

        res = self.sandbox.execute(code, verification_spec=contract)
        self.assertFalse(res["success"])
        self.assertIn("PostconditionVerificationError", res["error"])
        self.assertIn("has not been fitted", res["error"])

    def test_05_terminal_model_split_data_intent(self):
        """
        Verify that fitting a model on the full unsplit dataset X instead of X_train
        violates model training intent and raises PostconditionVerificationError.
        """
        code = """
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X, y)  # Intent violation: fitted on X instead of X_train!
var_final = clf
"""
        contract = VerificationContract(
            terminal_checks=[{
                "type": "model_fit_split",
                "cell_id": "SKLEARN.ENSEMBLE.RANDOMFORESTCLASSIFIER.FIT",
                "model_var": "clf",
                "feature_var": "X",
                "target_var": "y",
                "expected_train_feature_var": "X_train",
                "unsplit_feature_var": "X",
            }]
        )

        res = self.sandbox.execute(code, verification_spec=contract)
        self.assertFalse(res["success"])
        self.assertIn("PostconditionVerificationError", res["error"])
        self.assertIn("fitted on 'X' instead of split training partition 'X_train'", res["error"])
        self.assertIn("Model training intent violated", res["error"])

    def test_06_terminal_image_annotation_egress_success(self):
        """
        Verify that an image processing pipeline where drawn annotations are saved
        passes terminal intent verification.
        """
        # Create input image fixture
        raw_img_path = os.path.join(self.temp_dir, "raw_input.png")
        out_img_path = os.path.join(self.temp_dir, "annotated_output.png")
        raw_canvas = np.zeros((120, 120, 3), dtype=np.uint8)
        raw_canvas[30:90, 30:90] = 255
        cv2.imwrite(raw_img_path, raw_canvas)

        code = f"""
import cv2
import numpy as np

img = cv2.imread(r'{raw_img_path}')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
annotated = img.copy()
cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
saved = cv2.imwrite(r'{out_img_path}', annotated)
var_final = saved
"""
        contract = VerificationContract(
            terminal_checks=[{
                "type": "image_annotation_egress",
                "cell_id": "CV2_IMWRITE",
                "saved_var": "annotated",
                "annotated_var": "annotated",
                "ingress_var": "img",
                "output_path": f"'{out_img_path}'",
            }]
        )

        res = self.sandbox.execute(code, egress_paths=[out_img_path], verification_spec=contract)
        self.assertTrue(res["success"], f"Execution failed: {res.get('error')}")
        self.assertTrue(os.path.exists(out_img_path))
        # Ensure output is not identical to input
        saved_disk = cv2.imread(out_img_path)
        self.assertFalse(np.array_equal(saved_disk, raw_canvas), "Output should differ from raw input!")

    def test_07_terminal_image_unannotated_raw_egress_failure(self):
        """
        Verify that saving the unannotated raw input image instead of the annotated image
        raises PostconditionVerificationError.
        """
        raw_img_path = os.path.join(self.temp_dir, "raw_input_fail.png")
        out_img_path = os.path.join(self.temp_dir, "raw_egress_output.png")
        raw_canvas = np.zeros((120, 120, 3), dtype=np.uint8)
        raw_canvas[40:80, 40:80] = 200
        cv2.imwrite(raw_img_path, raw_canvas)

        # Pipeline draws contours on 'annotated', but saves 'img' (wire intent violation)
        code = f"""
import cv2
import numpy as np

img = cv2.imread(r'{raw_img_path}')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
annotated = img.copy()
cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
saved = cv2.imwrite(r'{out_img_path}', img)  # Intent violation: saved raw img instead of annotated!
var_final = saved
"""
        contract = VerificationContract(
            terminal_checks=[{
                "type": "image_annotation_egress",
                "cell_id": "CV2_IMWRITE",
                "saved_var": "img",
                "annotated_var": "annotated",
                "ingress_var": "img",
                "output_path": f"'{out_img_path}'",
            }]
        )

        res = self.sandbox.execute(code, egress_paths=[out_img_path], verification_spec=contract)
        self.assertFalse(res["success"])
        self.assertIn("PostconditionVerificationError", res["error"])
        self.assertIn("saved unannotated raw input image 'img' instead of annotated image 'annotated'", res["error"])

    def test_08_repair_cycle_with_postcondition_signal(self):
        """
        Verify that GEVRSandbox.repair_cycle supplies PostconditionVerificationError
        to the repair function, enabling automated one-shot correction of incorrect egress wiring.
        """
        raw_img_path = os.path.join(self.temp_dir, "raw_repair.png")
        out_img_path = os.path.join(self.temp_dir, "repaired_output.png")
        raw_canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        raw_canvas[20:60, 20:60] = 255
        cv2.imwrite(raw_img_path, raw_canvas)

        buggy_code = f"""
import cv2
import numpy as np

img = cv2.imread(r'{raw_img_path}')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
annotated = img.copy()
cv2.drawContours(annotated, contours, -1, (255, 0, 0), 2)
saved = cv2.imwrite(r'{out_img_path}', img)
var_final = saved
"""
        contract = VerificationContract(
            terminal_checks=[{
                "type": "image_annotation_egress",
                "cell_id": "CV2_IMWRITE",
                "saved_var": "img",
                "annotated_var": "annotated",
                "ingress_var": "img",
                "output_path": f"'{out_img_path}'",
            }]
        )

        repair_called = False
        received_error = ""

        def mock_llm_repair(code_to_fix: str, err_msg: str) -> str:
            nonlocal repair_called, received_error
            repair_called = True
            received_error = err_msg
            # The repair function diagnoses the error signal:
            if "saved unannotated raw input image 'img' instead of annotated image 'annotated'" in err_msg:
                # Update contract saved_var for subsequent pass
                contract.terminal_checks[0]["saved_var"] = "annotated"
                return code_to_fix.replace(f"cv2.imwrite(r'{out_img_path}', img)", f"cv2.imwrite(r'{out_img_path}', annotated)")
            return code_to_fix

        success, final_code, final_err = self.sandbox.repair_cycle(
            buggy_code,
            llm_repair_func=mock_llm_repair,
            max_attempts=2,
            egress_paths=[out_img_path],
            verification_spec=contract
        )

        self.assertTrue(repair_called, "Repair function was not invoked!")
        self.assertIn("PostconditionVerificationError", received_error)
        self.assertTrue(success, f"Repair cycle failed: {final_err}")
        self.assertIn(f"cv2.imwrite(r'{out_img_path}', annotated)", final_code)
        self.assertTrue(os.path.exists(out_img_path))

    def test_09_tabular_and_visualization_egress_verification(self):
        """
        Verify tabular egress (NaN detection in saved CSV) and visualization egress (empty canvas detection).
        """
        # A. Tabular egress with NaNs violating dropna intent
        csv_out_path = os.path.join(self.temp_dir, "egress_nan.csv")
        code_tabular_fail = f"""
import pandas as pd
import numpy as np

raw_df = pd.DataFrame({{'colA': [1, 2, np.nan], 'colB': [4.0, np.nan, 6.0]}})
raw_df.to_csv(r'{csv_out_path}', index=False)
var_final = True
"""
        contract_tab = VerificationContract(
            terminal_checks=[{
                "type": "tabular_egress",
                "cell_id": "PANDAS.DATAFRAME.TO_CSV",
                "saved_var": "raw_df",
                "ingress_var": "raw_df",
                "output_path": f"'{csv_out_path}'",
                "expected_clean": {"no_nans": True, "is_deduped": False}
            }]
        )
        res_tab = self.sandbox.execute(code_tabular_fail, egress_paths=[csv_out_path], verification_spec=contract_tab)
        self.assertFalse(res_tab["success"])
        self.assertIn("PostconditionVerificationError", res_tab["error"])
        self.assertTrue("contains NaN values" in res_tab["error"] or "saved raw uncleaned" in res_tab["error"])

        # B. Visualization egress: empty canvas with 0 plotted elements
        plot_out_path = os.path.join(self.temp_dir, "empty_plot.png")
        code_plot_fail = f"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# Zero plotting calls on ax!
fig.savefig(r'{plot_out_path}')
var_final = True
"""
        contract_plot = VerificationContract(
            terminal_checks=[{
                "type": "visualization_egress",
                "cell_id": "MATPLOTLIB.PYPLOT.SAVEFIG",
                "fig_var": "fig",
                "output_path": f"'{plot_out_path}'"
            }]
        )
        res_plot = self.sandbox.execute(code_plot_fail, egress_paths=[plot_out_path], verification_spec=contract_plot)
        self.assertFalse(res_plot["success"])
        self.assertIn("PostconditionVerificationError", res_plot["error"])
        self.assertIn("saved an empty figure", res_plot["error"])
        self.assertIn("0 plotted data elements", res_plot["error"])

    def test_10_unification_gate_end_to_end_contract_emission(self):
        """
        Verify that UnificationGate emits code and builds a complete VerificationContract
        from real cells in trees/lattice.db, and that the resulting pipeline executes and
        verifies successfully in the sandbox.
        """
        gate = UnificationGate()
        prompt = "Create synthetic dataset, split train and test, and fit a random forest model"
        ctx = ExecutionContext(prompt=prompt)

        cells = [
            self.orchestrator.loaded_cells["SKLEARN.DATASETS.MAKE_CLASSIFICATION"],
            self.orchestrator.loaded_cells["SKLEARN.MODEL_SELECTION.TRAIN_TEST_SPLIT"],
            self.orchestrator.loaded_cells["SKLEARN.ENSEMBLE.RANDOMFORESTCLASSIFIER.FIT"]
        ]

        code = gate.emit_code(cells, ctx)
        contract = gate.last_verification_contract

        self.assertIsNotNone(contract, "UnificationGate failed to generate a VerificationContract!")
        self.assertGreater(len(contract.cell_checks) + len(contract.terminal_checks), 0)

        # Verify terminal check structure
        term_check = next((tc for tc in contract.terminal_checks if tc.get("type") == "model_fit_split"), None)
        self.assertIsNotNone(term_check, "Terminal model_fit_split check not found in contract!")
        self.assertEqual(term_check["expected_train_feature_var"], "var_3")

        # Execute synthesized code with the verification contract
        res = self.sandbox.execute(code, verification_spec=contract)
        self.assertTrue(res["success"], f"End-to-end synthesized code failed verification:\n{res.get('error')}\nCode:\n{code}")
        self.assertEqual(res.get("error"), "")


if __name__ == "__main__":
    unittest.main()
