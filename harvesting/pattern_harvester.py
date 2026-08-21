# pattern_harvester.py
"""
Automated Algorithmic & Control-Flow Pattern Harvester for NSTL Engine.
Indexes core Python patterns, function definitions, binary operations,
and graph algorithms into the SQLite lattice database.
"""

import json
import sqlite3
import os
from typing import List, Dict, Any

CORE_CODE_PATTERNS: List[Dict[str, Any]] = [
    {
        "cell_id": "PYTHON_DEF_BINARY_FUNCTION",
        "domain_name": "python_core",
        "node_type": "pattern",
        "stage": 2,
        "keywords": ["function", "def", "add", "subtract", "multiply", "divide", "binary", "operator"],
        "input_type": "tuple",
        "input_state": "raw",
        "output_type": "int",
        "output_state": "computed",
        "code": "def add(a, b):\n    return a + b\n\n{output_var} = add(*{input_var})\nprint({output_var})",
        "dependencies": []
    },
    # Pandas ETL Pipeline Cells (_DEFAULT & _CELL aliases)
    {
        "cell_id": "PANDAS_READ_CSV_DEFAULT",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["csv", "read", "read_csv", "pandas", "load", "dataframe", "ingest"],
        "input_type": "str",
        "input_state": "source_identifier",
        "output_type": "DataFrame",
        "output_state": "raw",
        "code": "import pandas as pd\n\n{output_var} = pd.read_csv({input_var})",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_READ_CSV_CELL",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["csv", "read", "read_csv", "pandas", "load", "dataframe", "ingest"],
        "input_type": "str",
        "input_state": "source_identifier",
        "output_type": "DataFrame",
        "output_state": "raw",
        "code": "import pandas as pd\n\n{output_var} = pd.read_csv({input_var})",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_DROPNA_DEFAULT",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["dropna", "drop", "missing", "na", "null", "clean"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "cleaned",
        "code": "{output_var} = {input_var}.dropna()",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_DROPNA_CELL",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["dropna", "drop", "missing", "na", "null", "clean"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "cleaned",
        "code": "{output_var} = {input_var}.dropna()",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_DROP_DUPLICATES_DEFAULT",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["drop_duplicates", "duplicates", "dedup", "unique", "clean"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "deduplicated",
        "code": "{output_var} = {input_var}.drop_duplicates()",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_SORT_VALUES_DEFAULT",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["sort", "sort_values", "order", "ascending", "by"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "sorted",
        "code": "{output_var} = {input_var}.sort_values(by={by_column}, ascending={ascending})",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_SORT_VALUES_CELL",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["sort", "sort_values", "order", "ascending", "by"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "sorted",
        "code": "{output_var} = {input_var}.sort_values(by={by_column}, ascending={ascending})",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_TO_PARQUET_DEFAULT",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["to_parquet", "parquet", "write", "export", "save"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "str",
        "output_state": "filepath_written",
        "code": "{output_var} = {output_filename}\n{input_var}.to_parquet({output_filename})",
        "dependencies": ["pandas", "pyarrow"]
    },
    # Scikit-Learn ML Classification Pipeline Cells
    {
        "cell_id": "SKLEARN_SELECT_NUMERIC",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["select_dtypes", "numeric", "features", "pandas"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "numeric",
        "code": "{output_var} = {input_var}.select_dtypes(include=['number'])",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "SKLEARN_SELECT_NUMERIC_CELL",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["select_dtypes", "numeric", "features", "pandas"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "numeric",
        "code": "{output_var} = {input_var}.select_dtypes(include=['number'])",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "SKLEARN_STANDARD_SCALER",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["standardscaler", "scaler", "normalize", "scale"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "ndarray",
        "output_state": "scaled",
        "code": "from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\n{output_var} = scaler.fit_transform({input_var})",
        "dependencies": ["sklearn"]
    },
    {
        "cell_id": "SKLEARN_STANDARD_SCALER_CELL",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["standardscaler", "scaler", "normalize", "scale"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "ndarray",
        "output_state": "scaled",
        "code": "from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\n{output_var} = scaler.fit_transform({input_var})",
        "dependencies": ["sklearn"]
    },
    {
        "cell_id": "SKLEARN_TRAIN_TEST_SPLIT",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["train_test_split", "split", "partition"],
        "input_type": "ndarray",
        "input_state": "any",
        "output_type": "ndarray",
        "output_state": "split",
        "code": "from sklearn.model_selection import train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split({input_var}, {input_var}, test_size=0.2)\n{output_var} = X_train",
        "dependencies": ["sklearn"]
    },
    {
        "cell_id": "SKLEARN_RANDOM_FOREST_FIT",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["randomforestclassifier", "fit", "classifier", "train"],
        "input_type": "ndarray",
        "input_state": "any",
        "output_type": "Model",
        "output_state": "trained",
        "code": "from sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier()\nmodel.fit({input_var}, {input_var})\n{output_var} = model",
        "dependencies": ["sklearn"]
    },
    {
        "cell_id": "SKLEARN_RANDOM_FOREST_FIT_CELL",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["randomforestclassifier", "fit", "classifier", "train"],
        "input_type": "ndarray",
        "input_state": "any",
        "output_type": "Model",
        "output_state": "trained",
        "code": "from sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier()\nmodel.fit({input_var}, {input_var})\n{output_var} = model",
        "dependencies": ["sklearn"]
    },
    {
        "cell_id": "SKLEARN_MODEL_PREDICT",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["predict", "predictions", "model", "classifier"],
        "input_type": "Model",
        "input_state": "any",
        "output_type": "ndarray",
        "output_state": "predictions",
        "code": "{output_var} = {input_var}.predict({input_var})",
        "dependencies": ["sklearn"]
    },
    {
        "cell_id": "SKLEARN_MODEL_PREDICT_CELL",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["predict", "predictions", "model", "classifier"],
        "input_type": "Model",
        "input_state": "any",
        "output_type": "ndarray",
        "output_state": "predictions",
        "code": "{output_var} = {input_var}.predict({input_var})",
        "dependencies": ["sklearn"]
    },
    {
        "cell_id": "PANDAS_SAVE_PREDICTIONS",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["predictions", "to_csv", "save", "export"],
        "input_type": "ndarray",
        "input_state": "any",
        "output_type": "str",
        "output_state": "filepath_written",
        "code": "import pandas as pd\n\n_out_df = pd.DataFrame({input_var})\n{output_var} = _out_df.to_csv({output_filename}, index=False)",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_SAVE_PREDICTIONS_CELL",
        "domain_name": "machine_learning",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["predictions", "to_csv", "save", "export"],
        "input_type": "ndarray",
        "input_state": "any",
        "output_type": "str",
        "output_state": "filepath_written",
        "code": "import pandas as pd\n\n_out_df = pd.DataFrame({input_var})\n{output_var} = _out_df.to_csv({output_filename}, index=False)",
        "dependencies": ["pandas"]
    },
    # Graph & Pathfinding Cells
    {
        "cell_id": "PYTHON_GRAPH_BUILD_CELL",
        "domain_name": "algorithms",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["graph", "adjacency", "build", "network"],
        "input_type": "dict",
        "input_state": "any",
        "output_type": "Graph",
        "output_state": "adjacency_dict",
        "code": "{output_var} = {input_var}",
        "dependencies": []
    },
    {
        "cell_id": "PYTHON_ASTAR_PATHFINDING",
        "domain_name": "algorithms",
        "node_type": "algorithm",
        "stage": 2,
        "keywords": ["a*", "astar", "pathfinding", "shortest", "path", "heuristic", "heapq"],
        "input_type": "Graph",
        "input_state": "any",
        "output_type": "list",
        "output_state": "sorted",
        "code": (
            "import heapq\n\n"
            "def astar(graph, start, goal, h):\n"
            "    open_set = [(h(start, goal), 0, start, [start])]\n"
            "    visited = set()\n"
            "    while open_set:\n"
            "        f, g, current, path = heapq.heappop(open_set)\n"
            "        if current == goal:\n"
            "            return path\n"
            "        if current in visited:\n"
            "            continue\n"
            "        visited.add(current)\n"
            "        for neighbor, weight in graph.get(current, {}).items():\n"
            "            if neighbor not in visited:\n"
            "                heapq.heappush(open_set, (g + weight + h(neighbor, goal), g + weight, neighbor, path + [neighbor]))\n"
            "    return []\n\n"
            "{output_var} = astar({input_var}, start_node, goal_node, lambda a, b: 0)"
        ),
        "dependencies": ["heapq"]
    },
    {
        "cell_id": "PYTHON_DIJKSTRA_SHORTEST_PATH",
        "domain_name": "algorithms",
        "node_type": "algorithm",
        "stage": 2,
        "keywords": ["dijkstra", "shortest", "path", "graph", "heapq", "distances"],
        "input_type": "Graph",
        "input_state": "any",
        "output_type": "dict",
        "output_state": "distances",
        "code": (
            "import heapq\n\n"
            "def dijkstra(graph, start_node):\n"
            "    distances = {node: float('inf') for node in graph}\n"
            "    distances[start_node] = 0\n"
            "    pq = [(0, start_node)]\n"
            "    while pq:\n"
            "        curr_dist, curr_node = heapq.heappop(pq)\n"
            "        if curr_dist > distances[curr_node]:\n"
            "            continue\n"
            "        for neighbor, weight in graph[curr_node].items():\n"
            "            dist = curr_dist + weight\n"
            "            if dist < distances[neighbor]:\n"
            "                distances[neighbor] = dist\n"
            "                heapq.heappush(pq, (dist, neighbor))\n"
            "    return distances\n\n"
            "{output_var} = dijkstra({input_var}, start_node)"
        ),
        "dependencies": ["heapq"]
    },
    # NLP & Vector DB Cells
    {
        "cell_id": "SENTENCE_TRANSFORMER_ENCODE",
        "domain_name": "nlp",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["sentence", "transformer", "embed", "embedding", "vector", "encode"],
        "input_type": "str",
        "input_state": "any",
        "output_type": "ndarray",
        "output_state": "embeddings",
        "code": "from sentence_transformers import SentenceTransformer\n\nmodel = SentenceTransformer('all-MiniLM-L6-v2')\n{output_var} = model.encode([{input_var}])",
        "dependencies": ["sentence_transformers"]
    },
    {
        "cell_id": "FAISS_INDEX_ADD",
        "domain_name": "nlp",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["faiss", "index", "vector", "add", "store"],
        "input_type": "ndarray",
        "input_state": "any",
        "output_type": "FAISSIndex",
        "output_state": "indexed",
        "code": "import faiss\n\nd = {input_var}.shape[1]\nindex = faiss.IndexFlatL2(d)\nindex.add({input_var})\n{output_var} = index",
        "dependencies": ["faiss"]
    },
    {
        "cell_id": "FAISS_INDEX_SEARCH",
        "domain_name": "nlp",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["faiss", "search", "query", "nearest", "topk"],
        "input_type": "FAISSIndex",
        "input_state": "any",
        "output_type": "dict",
        "output_state": "search_results",
        "code": "import numpy as np\n\nD, I = {input_var}.search({input_var}, k=5)\n{output_var} = {'distances': D, 'indices': I}",
        "dependencies": ["faiss", "numpy"]
    },
    # EDA & Flask Cells
    {
        "cell_id": "PANDAS_DESCRIBE_DEFAULT",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["describe", "statistics", "summary", "pandas"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "summary",
        "code": "{output_var} = {input_var}.describe()",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_CORR_DEFAULT",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["corr", "correlation", "matrix", "pandas"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "DataFrame",
        "output_state": "correlation",
        "code": "{output_var} = {input_var}.corr()",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_TO_CSV_DEFAULT",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["to_csv", "export", "save", "csv"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "str",
        "output_state": "filepath_written",
        "code": "{output_var} = {output_filename}\n{input_var}.to_csv({output_filename}, index=False)",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "PANDAS_TO_CSV_CELL",
        "domain_name": "data_engineering",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["to_csv", "export", "save", "csv"],
        "input_type": "DataFrame",
        "input_state": "any",
        "output_type": "str",
        "output_state": "filepath_written",
        "code": "{output_var} = {output_filename}\n{input_var}.to_csv({output_filename}, index=False)",
        "dependencies": ["pandas"]
    },
    {
        "cell_id": "FLASK_CREATE_APP",
        "domain_name": "flask",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["flask", "app", "create"],
        "input_type": "str",
        "input_state": "source_identifier",
        "output_type": "Flask",
        "output_state": "app_created",
        "code": "from flask import Flask\n\n{output_var} = Flask(__name__)",
        "dependencies": ["flask"]
    },
    {
        "cell_id": "FLASK_SQLITE_CONNECT",
        "domain_name": "flask",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["sqlite", "connect", "database"],
        "input_type": "Flask",
        "input_state": "any",
        "output_type": "Flask",
        "output_state": "db_connected",
        "code": "import sqlite3\n\nconn = sqlite3.connect('app.db')\n{output_var} = {input_var}",
        "dependencies": ["flask", "sqlite3"]
    },
    {
        "cell_id": "FLASK_ADD_GET_ROUTE",
        "domain_name": "flask",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["route", "get", "endpoint"],
        "input_type": "Flask",
        "input_state": "any",
        "output_type": "Flask",
        "output_state": "routes_added",
        "code": "@ {input_var}.route('/api/data', methods=['GET'])\ndef get_data():\n    return {'status': 'ok'}\n{output_var} = {input_var}",
        "dependencies": ["flask"]
    },
    {
        "cell_id": "FLASK_ADD_POST_ROUTE",
        "domain_name": "flask",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["route", "post", "endpoint"],
        "input_type": "Flask",
        "input_state": "any",
        "output_type": "Flask",
        "output_state": "post_added",
        "code": "@ {input_var}.route('/api/data', methods=['POST'])\ndef create_data():\n    return {'created': True}\n{output_var} = {input_var}",
        "dependencies": ["flask"]
    },
    {
        "cell_id": "FLASK_ERROR_HANDLER",
        "domain_name": "flask",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["error", "handler", "404", "500"],
        "input_type": "Flask",
        "input_state": "any",
        "output_type": "Flask",
        "output_state": "handlers_added",
        "code": "@ {input_var}.errorhandler(404)\ndef not_found(e):\n    return {'error': 'not found'}, 404\n{output_var} = {input_var}",
        "dependencies": ["flask"]
    },
    {
        "cell_id": "FLASK_ENABLE_CORS",
        "domain_name": "flask",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["cors", "cross-origin"],
        "input_type": "Flask",
        "input_state": "any",
        "output_type": "Flask",
        "output_state": "cors_enabled",
        "code": "from flask_cors import CORS\n\nCORS({input_var})\n{output_var} = {input_var}",
        "dependencies": ["flask", "flask_cors"]
    },
    {
        "cell_id": "FLASK_RUN_DEBUG",
        "domain_name": "flask",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["run", "server", "start"],
        "input_type": "Flask",
        "input_state": "any",
        "output_type": "str",
        "output_state": "server_started",
        "code": "{output_var} = 'server_started'\n{input_var}.run(debug=True)",
        "dependencies": ["flask"]
    },
    {
        "cell_id": "CV2_IMREAD_CELL",
        "domain_name": "image_processing",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["cv2", "imread", "read", "image", "opencv", "load"],
        "input_type": "str",
        "input_state": "file_path",
        "output_type": "ndarray",
        "output_state": "bgr",
        "code": "import cv2\n\n{output_var} = cv2.imread({image_path})",
        "dependencies": ["cv2"]
    },
    {
        "cell_id": "CV2_CVTCOLOR_CELL",
        "domain_name": "image_processing",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["cv2", "cvtColor", "convert", "gray", "grayscale", "rgb", "hsv"],
        "input_type": "ndarray",
        "input_state": "any",
        "output_type": "ndarray",
        "output_state": "converted",
        "code": "import cv2\n\n{output_var} = cv2.cvtColor({input_var}, {code})",
        "dependencies": ["cv2"]
    },
    {
        "cell_id": "CV2_IMWRITE_CELL",
        "domain_name": "image_processing",
        "node_type": "micro",
        "stage": 2,
        "keywords": ["cv2", "imwrite", "write", "save", "image", "export"],
        "input_type": "ndarray",
        "input_state": "any",
        "output_type": "bool",
        "output_state": "saved",
        "code": "import cv2\n\n{output_var} = cv2.imwrite({output_path}, {input_var})",
        "dependencies": ["cv2"]
    }
]


def harvest_core_patterns(db_path: str):
    """Compiles algorithmic patterns directly into the SQLite lattice database."""
    if not db_path or not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            cell_id TEXT PRIMARY KEY,
            domain_name TEXT,
            node_type TEXT,
            stage INTEGER,
            keywords TEXT,
            input_type TEXT,
            input_state TEXT,
            output_type TEXT,
            output_state TEXT,
            code TEXT,
            dependencies TEXT,
            configuration_schema TEXT,
            verified INTEGER DEFAULT 1
        )
    """)

    for pattern in CORE_CODE_PATTERNS:
        cursor.execute(
            """
            INSERT OR REPLACE INTO nodes (
                cell_id, domain_name, node_type, stage, keywords,
                input_type, input_state, output_type, output_state,
                code, dependencies, configuration_schema, verified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern["cell_id"],
                pattern["domain_name"],
                pattern["node_type"],
                pattern["stage"],
                json.dumps(pattern["keywords"]),
                pattern["input_type"],
                pattern["input_state"],
                pattern["output_type"],
                pattern["output_state"],
                pattern["code"],
                json.dumps(pattern["dependencies"]),
                "{}",
                1
            )
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    db = os.path.join(os.path.dirname(__file__), "..", "trees", "nstl_lattice.db")
    harvest_core_patterns(db)
    print(f"Core patterns successfully harvested into {db}")
