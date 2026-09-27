import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lattice import LatticeOrchestrator
from internal_rag import LocalRAG
from router import LatticeRouter, HardwareProfiler
from inference import ModelManager

# Benchmark Labeled Test Dataset: (prompt, domain, expected_cell_id)
LABELED_DATASET: List[Dict[str, str]] = [
    # --- OPENCV (CV2) PROMPTS ---
    {"prompt": "convert image from BGR to grayscale using opencv", "domain": "cv2", "expected": "CV2_CVTCOLOR_COLOR_BGR2GRAY"},
    {"prompt": "convert BGR image to HSV color space in opencv", "domain": "cv2", "expected": "CV2_CVTCOLOR_COLOR_BGR2HSV"},
    {"prompt": "convert BGR photo to RGB format in opencv", "domain": "cv2", "expected": "CV2_CVTCOLOR_COLOR_BGR2RGB"},
    {"prompt": "convert BGR to YCrCb color space in cv2", "domain": "cv2", "expected": "CV2_CVTCOLOR_COLOR_BGR2YCRCB"},
    {"prompt": "apply binary thresholding to grayscale image", "domain": "cv2", "expected": "CV2_THRESHOLD_THRESH_BINARY"},
    {"prompt": "apply otsu automatic thresholding to image", "domain": "cv2", "expected": "CV2_THRESHOLD_THRESH_OTSU"},
    {"prompt": "resize image to 256x256 using cubic interpolation", "domain": "cv2", "expected": "CV2_RESIZE_INTER_CUBIC"},
    {"prompt": "resize frame using nearest neighbor interpolation", "domain": "cv2", "expected": "CV2_RESIZE_INTER_NEAREST"},
    {"prompt": "apply Canny edge detection filter to image", "domain": "cv2", "expected": "cv2.Canny"},
    {"prompt": "apply Gaussian blur to smooth the image", "domain": "cv2", "expected": "cv2.GaussianBlur"},
    {"prompt": "apply median blur to remove salt and pepper noise", "domain": "cv2", "expected": "cv2.medianBlur"},
    {"prompt": "find external contours in binary image", "domain": "cv2", "expected": "CV2_FINDCONTOURS_RETR_EXTERNAL"},
    {"prompt": "find all contour hierarchy trees in mask", "domain": "cv2", "expected": "CV2_FINDCONTOURS_RETR_TREE"},
    {"prompt": "perform morphological opening on binary mask", "domain": "cv2", "expected": "CV2_MORPHOLOGYEX_MORPH_OPEN"},
    {"prompt": "perform morphological closing to fill holes", "domain": "cv2", "expected": "CV2_MORPHOLOGYEX_MORPH_CLOSE"},
    {"prompt": "normalize image pixel values between 0 and 255 using minmax", "domain": "cv2", "expected": "CV2_NORMALIZE_NORM_MINMAX"},
    {"prompt": "normalize feature vector using L2 norm", "domain": "cv2", "expected": "CV2_NORMALIZE_NORM_L2"},
    {"prompt": "compute distance transform using Euclidean L2 distance", "domain": "cv2", "expected": "CV2_DISTANCETRANSFORM_DIST_L2"},
    {"prompt": "compute distance transform using Manhattan L1 distance", "domain": "cv2", "expected": "CV2_DISTANCETRANSFORM_DIST_L1"},
    {"prompt": "read image file from disk in color mode", "domain": "cv2", "expected": "cv2.imread"},
    {"prompt": "save processed image array to disk as jpeg", "domain": "cv2", "expected": "cv2.imwrite"},
    {"prompt": "rotate image 90 degrees clockwise in opencv", "domain": "cv2", "expected": "CV2_ROTATE_ROTATE_90_CLOCKWISE"},
    {"prompt": "warp affine transformation on image matrix", "domain": "cv2", "expected": "CV2_WARPAFFINE_INTER_LINEAR"},
    {"prompt": "compute optical flow using Farneback algorithm", "domain": "cv2", "expected": "CV2_FARNEBACKOPTICALFLOW_CREATE_ALGO_HINT_ACCURATE"},

    # --- PANDAS PROMPTS ---
    {"prompt": "load customer dataset from CSV file into pandas DataFrame", "domain": "pandas", "expected": "PANDAS_READ_GROUP_READ_CSV"},
    {"prompt": "load sales dataset from Parquet file into pandas", "domain": "pandas", "expected": "PANDAS_READ_GROUP_READ_PARQUET"},
    {"prompt": "load data from Excel file worksheet into pandas", "domain": "pandas", "expected": "PANDAS_READ_GROUP_READ_EXCEL"},
    {"prompt": "read JSON records into pandas DataFrame", "domain": "pandas", "expected": "PANDAS_READ_GROUP_READ_JSON"},
    {"prompt": "read SQL query result into pandas DataFrame", "domain": "pandas", "expected": "PANDAS_READ_GROUP_READ_SQL"},
    {"prompt": "drop rows with missing NA values from DataFrame", "domain": "pandas", "expected": "pd.DataFrame.dropna"},
    {"prompt": "fill missing null values with zero in DataFrame", "domain": "pandas", "expected": "pd.DataFrame.fillna"},
    {"prompt": "sort DataFrame rows by revenue column descending", "domain": "pandas", "expected": "pd.DataFrame.sort_values"},
    {"prompt": "group DataFrame by category and calculate sum", "domain": "pandas", "expected": "pd.DataFrame.groupby"},
    {"prompt": "merge two DataFrames on primary key column", "domain": "pandas", "expected": "pd.DataFrame.merge"},
    {"prompt": "concatenate list of DataFrames along rows", "domain": "pandas", "expected": "pd.concat"},
    {"prompt": "save DataFrame to CSV file without index", "domain": "pandas", "expected": "PANDAS_TO_GROUP_TO_PICKLE"},
    {"prompt": "convert column data type to datetime in pandas", "domain": "pandas", "expected": "PANDAS_TO_GROUP_TO_DATETIME"},
    {"prompt": "convert column values to numeric float type", "domain": "pandas", "expected": "PANDAS_TO_GROUP_TO_NUMERIC"},
    {"prompt": "compute summary statistics for all numerical columns", "domain": "pandas", "expected": "pd.DataFrame.describe"},
    {"prompt": "filter DataFrame rows where age is greater than 30", "domain": "pandas", "expected": "pd.DataFrame.query"},
    {"prompt": "rename columns in DataFrame using dictionary mapping", "domain": "pandas", "expected": "pd.DataFrame.rename"},
    {"prompt": "drop column from DataFrame by name", "domain": "pandas", "expected": "pd.DataFrame.drop"},
    {"prompt": "reset index of DataFrame after filtering", "domain": "pandas", "expected": "pd.DataFrame.reset_index"},
    {"prompt": "compute correlation matrix between numeric columns", "domain": "pandas", "expected": "pd.DataFrame.corr"},

    # --- NUMPY PROMPTS ---
    {"prompt": "create numpy array initialized with zeros", "domain": "numpy", "expected": "np.zeros"},
    {"prompt": "create numpy array filled with ones", "domain": "numpy", "expected": "np.ones"},
    {"prompt": "create evenly spaced numbers over specified interval", "domain": "numpy", "expected": "np.linspace"},
    {"prompt": "create array with range of numbers with step size", "domain": "numpy", "expected": "np.arange"},
    {"prompt": "reshape numpy array to specified 2D dimensions", "domain": "numpy", "expected": "np.reshape"},
    {"prompt": "transpose numpy 2D matrix axes", "domain": "numpy", "expected": "np.transpose"},
    {"prompt": "concatenate arrays along specified axis", "domain": "numpy", "expected": "np.concatenate"},
    {"prompt": "stack numpy arrays vertically row-wise", "domain": "numpy", "expected": "np.vstack"},
    {"prompt": "stack numpy arrays horizontally column-wise", "domain": "numpy", "expected": "np.hstack"},
    {"prompt": "compute mean value of numpy array elements", "domain": "numpy", "expected": "np.mean"},
    {"prompt": "compute standard deviation of numpy array", "domain": "numpy", "expected": "np.std"},
    {"prompt": "compute matrix multiplication dot product of two arrays", "domain": "numpy", "expected": "np.dot"},
    {"prompt": "find indices of elements where condition is true", "domain": "numpy", "expected": "np.where"},
    {"prompt": "find unique elements of an array", "domain": "numpy", "expected": "NUMPY_UNIQUE_GROUP_UNIQUE"},
    {"prompt": "compute argmax index of maximum value in array", "domain": "numpy", "expected": "np.argmax"},

    # --- SCIKIT-LEARN (SKLEARN) PROMPTS ---
    {"prompt": "scale feature matrix using StandardScaler zero mean unit variance", "domain": "sklearn", "expected": "SKLEARN_SCALER_FAMILY_STANDARDSCALER"},
    {"prompt": "scale features between 0 and 1 using MinMaxScaler", "domain": "sklearn", "expected": "SKLEARN_SCALER_FAMILY_MINMAXSCALER"},
    {"prompt": "scale features using RobustScaler robust to outliers", "domain": "sklearn", "expected": "SKLEARN_SCALER_FAMILY_ROBUSTSCALER"},
    {"prompt": "one-hot encode categorical features into binary vectors", "domain": "sklearn", "expected": "SKLEARN_ENCODER_FAMILY_ONEHOTENCODER"},
    {"prompt": "label encode categorical targets into integer labels", "domain": "sklearn", "expected": "SKLEARN_ENCODER_FAMILY_LABELENCODER"},
    {"prompt": "split dataset into train and test random subsets", "domain": "sklearn", "expected": "SKLEARN_SPLIT_FAMILY_TRAIN_TEST_SPLIT"},
    {"prompt": "fit RandomForestClassifier ensemble model on training data", "domain": "sklearn", "expected": "SKLEARN_CLASSIFIER_FAMILY_RANDOMFORESTCLASSIFIER"},
    {"prompt": "fit LogisticRegression binary classification model", "domain": "sklearn", "expected": "SKLEARN_CLASSIFIER_FAMILY_LOGISTICREGRESSION"},
    {"prompt": "fit Support Vector Machine SVC classifier on dataset", "domain": "sklearn", "expected": "SKLEARN_CLASSIFIER_FAMILY_SVC"},
    {"prompt": "fit GradientBoostingClassifier ensemble model", "domain": "sklearn", "expected": "SKLEARN_CLASSIFIER_FAMILY_GRADIENTBOOSTINGCLASSIFIER"},
    {"prompt": "predict class labels for test feature samples", "domain": "sklearn", "expected": "sklearn.predict"},
    {"prompt": "predict class probabilities for test samples", "domain": "sklearn", "expected": "sklearn.predict_proba"},

    # --- SCIPY PROMPTS ---
    {"prompt": "minimize scalar objective function using BFGS method", "domain": "scipy", "expected": "scipy.optimize.minimize"},
    {"prompt": "minimize constrained objective function using Nelder-Mead", "domain": "scipy", "expected": "scipy.optimize.minimize"},
    {"prompt": "compute cubic spline interpolation over 1D data points", "domain": "scipy", "expected": "SCIPY_SPLINE_FAMILY_CUBICSPLINE"},
    {"prompt": "solve initial value problem differential equation", "domain": "scipy", "expected": "scipy.integrate.solve_ivp"},

    # --- ALGORITHMS / MACROS PROMPTS ---
    {"prompt": "Dijkstra algorithm for shortest path on weighted graph", "domain": "algorithms", "expected": "MACRO_DIJKSTRA"},
    {"prompt": "A* search algorithm pathfinding on grid graph", "domain": "algorithms", "expected": "MACRO_ASTAR_PATHFINDING"},
    {"prompt": "end-to-end ETL data cleaning and transformation pipeline", "domain": "algorithms", "expected": "MACRO_ETL_PIPELINE"},
    {"prompt": "machine learning classification training pipeline with scaling", "domain": "algorithms", "expected": "MACRO_ML_CLASSIFICATION_PIPELINE"},
]


def evaluate_routing_accuracy():
    print("=== NSTL Routing & Retrieval Accuracy Harness ===")
    
    # 1. Initialize Pipeline components on CPU to prevent CUDA OOM
    HardwareProfiler.set_config("cuda", "cuda", "vram")
    ModelManager.get_instance().initialize_profile("E")
    orchestrator = LatticeOrchestrator("trees")
    rag = LocalRAG(trees_dir="trees", orchestrator=orchestrator)
    router = LatticeRouter(orchestrator=orchestrator, rag_engine=rag)

    all_cells = orchestrator.get_all_available_cells()
    print(f"[+] Total Available Cells in Orchestrator: {len(all_cells)}")
    print(f"[+] FAISS Index Size: {rag.index.ntotal}")

    top1_hits = 0
    top5_hits = 0
    total_evals = len(LABELED_DATASET)

    domain_stats: Dict[str, Dict[str, int]] = {}
    detailed_results = []

    for idx, item in enumerate(LABELED_DATASET):
        prompt = item["prompt"]
        domain = item["domain"]
        expected = item["expected"]

        domain_stats.setdefault(domain, {"total": 0, "top1": 0, "top5": 0})
        domain_stats[domain]["total"] += 1

        emb = ModelManager.get_instance().get_embeddings([prompt])[0]
        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        elif not isinstance(emb, np.ndarray):
            emb = np.array(emb, dtype=np.float32)

        norm_val = np.linalg.norm(emb)
        emb = emb / (norm_val if norm_val > 0 else 1.0)
        emb_arr = np.array([emb], dtype=np.float32)

        # Retrieve scored candidates
        scored = router._score_candidates(all_cells, emb_arr, goal=prompt)
        candidate_ids = [cell.cell_id for _, cell in scored[:10]]

        top1_match = False
        top5_match = False

        if candidate_ids:
            # Check Top-1 match (fuzzy or exact ID prefix)
            if candidate_ids[0].lower().startswith(expected.lower()) or expected.lower() in candidate_ids[0].lower():
                top1_match = True
                top1_hits += 1
                domain_stats[domain]["top1"] += 1

            # Check Top-5 match
            for cid in candidate_ids[:5]:
                if cid.lower().startswith(expected.lower()) or expected.lower() in cid.lower():
                    top5_match = True
                    top5_hits += 1
                    domain_stats[domain]["top5"] += 1
                    break

        detailed_results.append({
            "prompt": prompt,
            "domain": domain,
            "expected": expected,
            "top1_match": top1_match,
            "top5_match": top5_match,
            "top1_retrieved": candidate_ids[0] if candidate_ids else "NONE",
            "top5_retrieved": candidate_ids[:5]
        })

    top1_acc = (top1_hits / total_evals) * 100.0
    top5_acc = (top5_hits / total_evals) * 100.0

    print("\n" + "="*80)
    print(f" ROUTING ACCURACY EVALUATION RESULTS ({total_evals} Total Evaluated Prompts)")
    print("="*80)
    print(f" Overall Top-1 Accuracy: {top1_acc:.2f}% ({top1_hits}/{total_evals})")
    print(f" Overall Top-5 Accuracy: {top5_acc:.2f}% ({top5_hits}/{total_evals})")
    print("-" * 80)
    print(" Breakdown by Domain:")

    domain_summary = {}
    for d, s in domain_stats.items():
        d_top1 = (s["top1"] / s["total"]) * 100.0 if s["total"] > 0 else 0.0
        d_top5 = (s["top5"] / s["total"]) * 100.0 if s["total"] > 0 else 0.0
        print(f"   * Domain {d:<12} -> Top-1: {d_top1:6.2f}% | Top-5: {d_top5:6.2f}% ({s['total']} prompts)")
        domain_summary[d] = {
            "total_prompts": s["total"],
            "top1_accuracy": d_top1,
            "top5_accuracy": d_top5
        }

    print("="*80)

    # Save baseline report to logs/routing_accuracy_baseline.json
    baseline_file = os.path.abspath(os.path.join("logs", "routing_accuracy_baseline.json"))
    os.makedirs(os.path.dirname(baseline_file), exist_ok=True)

    baseline_data = {
        "overall_top1_accuracy": top1_acc,
        "overall_top5_accuracy": top5_acc,
        "total_prompts_evaluated": total_evals,
        "domain_summary": domain_summary,
        "detailed_evaluations": detailed_results
    }

    print(f"[DEBUG WRITE] Writing baseline report to: {baseline_file}")
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"[+] Saved baseline accuracy report to: {baseline_file}")


if __name__ == "__main__":
    evaluate_routing_accuracy()
