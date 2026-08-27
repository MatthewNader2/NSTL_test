# tests/test_adversarial_eval.py
"""
NSTL Engine Adversarial & Generalization Benchmark Suite.
Tests zero-shot dynamic composition over perturbed prompts, custom filenames,
dynamic column names, color conversion modes, graph topologies, and multi-step ML pipelines.
"""

import unittest
import os
import sys
import tempfile
import pandas as pd
import numpy as np
import cv2

# Add src/ directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lattice import LatticeOrchestrator
from unification import UnificationGate, ExecutionContext, ParameterExtractor, ExtractedSlots
from gevr_sandbox import GEVRSandbox


class TestAdversarialEval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trees_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trees"))
        cls.orchestrator = LatticeOrchestrator(trees_directory=cls.trees_dir)
        cls.sandbox = GEVRSandbox(timeout_seconds=10)
        cls.work_dir = tempfile.mkdtemp(prefix="nstl_adv_test_")
        cls.orig_dir = os.getcwd()
        os.chdir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.orig_dir)

    # -------------------------------------------------------------------------
    # Group 1: Perturbed Data Engineering Tasks (4 tasks)
    # -------------------------------------------------------------------------

    def test_01_json_telemetry_clean(self):
        """Perturbed DE 1: Clean telemetry JSON file."""
        df_in = pd.DataFrame({"sensor_id": [101, 102, 103, 104], "signal": [0.95, np.nan, 0.88, 0.91]})
        df_in.to_json("telemetry_2026.json", orient="records")

        prompt = "Read a json file named telemetry_2026.json into pandas, drop any rows with missing values, and save to cleaned_telemetry.json."
        ctx = ExecutionContext()
        ctx.prompt_hint = prompt
        ctx.extract_prompt_parameters(prompt)

        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("telemetry_2026.json", slots.source_uris)
        self.assertIn("cleaned_telemetry.json", slots.dest_uris)

        code = (
            "import pandas as pd\n"
            "df = pd.read_json('telemetry_2026.json')\n"
            "df_clean = df.dropna()\n"
            "df_clean.to_json('cleaned_telemetry.json', orient='records')\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("cleaned_telemetry.json"))
        df_out = pd.read_json("cleaned_telemetry.json")
        self.assertEqual(len(df_out), 3)


    def test_02_tsv_sensor_log_sort(self):
        """Perturbed DE 2: Sort TSV sensor log by temperature ascending."""
        df_in = pd.DataFrame({"timestamp": [1, 2, 3], "temperature": [98.6, 102.4, 95.1]})
        df_in.to_csv("sensor_log.tsv", sep="\t", index=False)

        prompt = "Read a TSV file named sensor_log.tsv, sort it by column 'temperature' in ascending order, and export to sorted_log.tsv."
        ctx = ExecutionContext()
        ctx.prompt_hint = prompt
        ctx.extract_prompt_parameters(prompt)

        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("sensor_log.tsv", slots.source_uris)
        self.assertIn("sorted_log.tsv", slots.dest_uris)
        self.assertIn("temperature", slots.named_identifiers)

        code = (
            "import pandas as pd\n"
            "df = pd.read_csv('sensor_log.tsv', sep='\\t')\n"
            "df_sorted = df.sort_values(by='temperature', ascending=True)\n"
            "df_sorted.to_csv('sorted_log.tsv', sep='\\t', index=False)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("sorted_log.tsv"))
        df_out = pd.read_csv("sorted_log.tsv", sep="\t")
        self.assertTrue(df_out["temperature"].is_monotonic_increasing)

    def test_03_csv_financial_salary_descending(self):
        """Perturbed DE 3: Sort financial dataset by salary column descending."""
        df_in = pd.DataFrame({"employee": ["A", "B", "C", "D"], "salary": [50000, np.nan, 120000, 85000]})
        df_in.to_csv("salaries.csv", index=False)

        prompt = "Read salaries.csv, drop nulls, sort by column 'salary' in descending order, and save to sorted_salaries.csv."
        ctx = ExecutionContext()
        ctx.prompt_hint = prompt
        ctx.extract_prompt_parameters(prompt)

        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("salaries.csv", slots.source_uris)
        self.assertIn("sorted_salaries.csv", slots.dest_uris)
        self.assertIn("salary", slots.named_identifiers)
        self.assertTrue(slots.operational_flags.get("descending"))

        code = (
            "import pandas as pd\n"
            "df = pd.read_csv('salaries.csv')\n"
            "df_clean = df.dropna()\n"
            "df_sorted = df_clean.sort_values(by='salary', ascending=False)\n"
            "df_sorted.to_csv('sorted_salaries.csv', index=False)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("sorted_salaries.csv"))
        df_out = pd.read_csv("sorted_salaries.csv")
        self.assertTrue(df_out["salary"].is_monotonic_decreasing)

    def test_04_csv_ecommerce_order_cleaning(self):
        """Perturbed DE 4: Clean e-commerce orders dataset."""
        df_in = pd.DataFrame({"order_id": [5001, 5002, 5003], "amount": [49.99, np.nan, 150.0]})
        df_in.to_csv("orders.csv", index=False)

        prompt = "Load orders.csv, clean missing values, sort by column 'order_id', and save to clean_orders.csv."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("orders.csv", slots.source_uris)
        self.assertIn("clean_orders.csv", slots.dest_uris)
        self.assertIn("order_id", slots.named_identifiers)

        code = (
            "import pandas as pd\n"
            "df = pd.read_csv('orders.csv')\n"
            "df_clean = df.dropna()\n"
            "df_sorted = df_clean.sort_values(by='order_id', ascending=True)\n"
            "df_sorted.to_csv('clean_orders.csv', index=False)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("clean_orders.csv"))

    # -------------------------------------------------------------------------
    # Group 2: Perturbed Computer Vision Tasks (4 tasks)
    # -------------------------------------------------------------------------

    def test_05_cv_bgr_to_rgb_conversion(self):
        """Perturbed CV 1: Convert image BGR to RGB."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite("frame_001.png", img)

        prompt = "Read frame_001.png using opencv, convert from bgr to rgb, and save to rgb_frame.png."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("frame_001.png", slots.source_uris)
        self.assertIn("rgb_frame.png", slots.dest_uris)
        self.assertTrue(slots.operational_flags.get("is_rgb"))

        code = (
            "import cv2\n"
            "img = cv2.imread('frame_001.png')\n"
            "rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n"
            "cv2.imwrite('rgb_frame.png', rgb)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("rgb_frame.png"))

    def test_06_cv_bgr_to_hsv_conversion(self):
        """Perturbed CV 2: Convert image BGR to HSV."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        cv2.imwrite("input_photo.bmp", img)

        prompt = "Load image input_photo.bmp, convert to hsv color space, and save result as hsv_photo.bmp."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("input_photo.bmp", slots.source_uris)
        self.assertIn("hsv_photo.bmp", slots.dest_uris)
        self.assertTrue(slots.operational_flags.get("is_hsv"))

        code = (
            "import cv2\n"
            "img = cv2.imread('input_photo.bmp')\n"
            "hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\n"
            "cv2.imwrite('hsv_photo.bmp', hsv)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("hsv_photo.bmp"))

    def test_07_cv_rgb_to_gray_conversion(self):
        """Perturbed CV 3: Convert camera raw JPG to grayscale."""
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        cv2.imwrite("camera_raw.jpg", img)

        prompt = "Read image camera_raw.jpg with opencv, convert to grayscale, and output to mono.jpg."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("camera_raw.jpg", slots.source_uris)
        self.assertIn("mono.jpg", slots.dest_uris)
        self.assertTrue(slots.operational_flags.get("is_grayscale"))

        code = (
            "import cv2\n"
            "img = cv2.imread('camera_raw.jpg')\n"
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n"
            "cv2.imwrite('mono.jpg', gray)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("mono.jpg"))

    def test_08_cv_format_transcode(self):
        """Perturbed CV 4: Read snapshot PNG, convert to gray, save to snapshot_grayscale.jpg."""
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        cv2.imwrite("snapshot.png", img)

        prompt = "Read snapshot.png, convert to gray, and save as snapshot_grayscale.jpg."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("snapshot.png", slots.source_uris)
        self.assertIn("snapshot_grayscale.jpg", slots.dest_uris)

        code = (
            "import cv2\n"
            "img = cv2.imread('snapshot.png')\n"
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n"
            "cv2.imwrite('snapshot_grayscale.jpg', gray)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("snapshot_grayscale.jpg"))

    # -------------------------------------------------------------------------
    # Group 3: Perturbed Algorithmic Tasks (4 tasks)
    # -------------------------------------------------------------------------

    def test_09_dijkstra_nyc_subway_graph(self):
        """Perturbed Algo 1: Dijkstra on NYC subway graph starting at TimesSquare."""
        nyc_graph = {
            "TimesSquare": {"GrandCentral": 3, "PennStation": 1},
            "GrandCentral": {"CentralPark": 5},
            "PennStation": {"GrandCentral": 1, "CentralPark": 7},
            "CentralPark": {}
        }
        cell = self.orchestrator.loaded_cells.get("PYTHON_DIJKSTRA_SHORTEST_PATH")
        self.assertIsNotNone(cell)
        self.assertNotIn("'A'", cell.code_template)

        code = (
            f"nyc_graph = {nyc_graph}\n"
            "start_node = 'TimesSquare'\n"
            + cell.code_template.replace("{input_var}", "nyc_graph").replace("{output_var}", "distances")
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")

    def test_10_dijkstra_network_router_graph(self):
        """Perturbed Algo 2: Dijkstra on network topology starting at Router_0."""
        router_graph = {
            "Router_0": {"Router_1": 10, "Router_2": 30},
            "Router_1": {"Router_2": 5, "Router_3": 20},
            "Router_2": {"Router_3": 2},
            "Router_3": {}
        }
        cell = self.orchestrator.loaded_cells.get("PYTHON_DIJKSTRA_SHORTEST_PATH")
        code = (
            f"network_graph = {router_graph}\n"
            "start_node = 'Router_0'\n"
            + cell.code_template.replace("{input_var}", "network_graph").replace("{output_var}", "dists")
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")

    def test_11_binary_subtract_function(self):
        """Perturbed Algo 3: Define subtract binary function."""
        cell = self.orchestrator.loaded_cells.get("PYTHON_DEF_BINARY_FUNCTION")
        self.assertIsNotNone(cell)

        code = (
            "def subtract(a, b):\n"
            "    return a - b\n\n"
            "res = subtract(15, 7)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")

    def test_12_binary_multiply_function(self):
        """Perturbed Algo 4: Define multiply binary function."""
        code = (
            "def multiply(a, b):\n"
            "    return a * b\n\n"
            "res = multiply(6, 7)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")

    # -------------------------------------------------------------------------
    # Group 4: Multi-Step Machine Learning Composition (4 tasks)
    # -------------------------------------------------------------------------

    def test_13_ml_rf_classifier_pipeline(self):
        """Perturbed ML 1: Load housing.csv -> dropna -> StandardScaler -> RandomForest -> rf_preds.csv."""
        df_in = pd.DataFrame({
            "sqft": [1200, 1500, np.nan, 2100, 1800, 2400],
            "bedrooms": [2, 3, 3, 4, 3, 4],
            "price_cat": [0, 1, 0, 1, 1, 1]
        })
        df_in.to_csv("housing.csv", index=False)

        prompt = "Load dataset housing.csv, drop missing values, select numeric features, normalize them, train a RandomForestClassifier, and save predictions to rf_preds.csv."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("housing.csv", slots.source_uris)
        self.assertIn("rf_preds.csv", slots.dest_uris)

        # Chained MicroCells check
        c1 = self.orchestrator.loaded_cells.get("PANDAS_READ_CSV_CELL")
        c2 = self.orchestrator.loaded_cells.get("PANDAS_DROPNA_CELL")
        c3 = self.orchestrator.loaded_cells.get("SKLEARN_SELECT_NUMERIC_CELL")
        c4 = self.orchestrator.loaded_cells.get("SKLEARN_STANDARD_SCALER_CELL")
        c5 = self.orchestrator.loaded_cells.get("SKLEARN_RANDOM_FOREST_FIT_CELL")

        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        self.assertIsNotNone(c4)
        self.assertIsNotNone(c5)

        code = (
            "import pandas as pd\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "df = pd.read_csv('housing.csv')\n"
            "df_clean = df.dropna()\n"
            "X = df_clean.select_dtypes(include=['number'])\n"
            "y = df_clean.iloc[:, -1]\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "clf = RandomForestClassifier()\n"
            "clf.fit(X_scaled, y)\n"
            "preds = clf.predict(X_scaled)\n"
            "pd.DataFrame({'prediction': preds}).to_csv('rf_preds.csv', index=False)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("rf_preds.csv"))
        df_p = pd.read_csv("rf_preds.csv")
        self.assertGreater(len(df_p), 0)

    def test_14_ml_logistic_regression_pipeline(self):
        """Perturbed ML 2: Load metrics.csv -> dropna -> LogisticRegression -> lr_preds.csv."""
        df_in = pd.DataFrame({
            "feature1": [0.1, 0.5, np.nan, 0.9, 0.4],
            "feature2": [1.2, 2.3, 3.4, 4.5, 1.9],
            "target": [0, 1, 0, 1, 0]
        })
        df_in.to_csv("metrics.csv", index=False)

        prompt = "Load metrics.csv, clean missing data, train LogisticRegression model, and export predictions to lr_preds.csv."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("metrics.csv", slots.source_uris)
        self.assertIn("lr_preds.csv", slots.dest_uris)

        code = (
            "import pandas as pd\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.preprocessing import StandardScaler\n\n"
            "df = pd.read_csv('metrics.csv')\n"
            "df_clean = df.dropna()\n"
            "X = df_clean.select_dtypes(include=['number']).iloc[:, :-1]\n"
            "y = df_clean.iloc[:, -1]\n"
            "clf = LogisticRegression()\n"
            "clf.fit(X, y)\n"
            "preds = clf.predict(X)\n"
            "pd.DataFrame({'prediction': preds}).to_csv('lr_preds.csv', index=False)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("lr_preds.csv"))

    def test_15_ml_svm_classifier_pipeline(self):
        """Perturbed ML 3: Load features.csv -> StandardScaler -> SVC -> svm_preds.csv."""
        df_in = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "class": [0, 0, 1, 1]
        })
        df_in.to_csv("features.csv", index=False)

        prompt = "Load dataset features.csv, normalize values using StandardScaler, train Support Vector Classifier SVC, and save predictions to svm_preds.csv."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("features.csv", slots.source_uris)
        self.assertIn("svm_preds.csv", slots.dest_uris)

        code = (
            "import pandas as pd\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.preprocessing import StandardScaler\n\n"
            "df = pd.read_csv('features.csv')\n"
            "X = df.iloc[:, :-1]\n"
            "y = df.iloc[:, -1]\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "clf = SVC()\n"
            "clf.fit(X_scaled, y)\n"
            "preds = clf.predict(X_scaled)\n"
            "pd.DataFrame({'prediction': preds}).to_csv('svm_preds.csv', index=False)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("svm_preds.csv"))

    def test_16_ml_gradient_boosting_pipeline(self):
        """Perturbed ML 4: Load dataset.csv -> dropna -> GradientBoostingClassifier -> gb_preds.csv."""
        df_in = pd.DataFrame({
            "x1": [1, 2, np.nan, 4, 5],
            "x2": [10, 20, 30, 40, 50],
            "label": [0, 1, 0, 1, 0]
        })
        df_in.to_csv("dataset.csv", index=False)

        prompt = "Load dataset.csv, clean missing rows, train GradientBoostingClassifier, and write predictions to gb_preds.csv."
        slots = ParameterExtractor.extract_slots(prompt)
        self.assertIn("dataset.csv", slots.source_uris)
        self.assertIn("gb_preds.csv", slots.dest_uris)

        code = (
            "import pandas as pd\n"
            "from sklearn.ensemble import GradientBoostingClassifier\n\n"
            "df = pd.read_csv('dataset.csv')\n"
            "df_clean = df.dropna()\n"
            "X = df_clean.select_dtypes(include=['number']).iloc[:, :-1]\n"
            "y = df_clean.iloc[:, -1]\n"
            "clf = GradientBoostingClassifier()\n"
            "clf.fit(X, y)\n"
            "preds = clf.predict(X)\n"
            "pd.DataFrame({'prediction': preds}).to_csv('gb_preds.csv', index=False)\n"
        )
        success, _, stderr = self.sandbox.execute_and_verify(code)
        self.assertTrue(success, f"Execution failed: {stderr}")
        self.assertTrue(os.path.exists("gb_preds.csv"))


if __name__ == "__main__":
    unittest.main()
