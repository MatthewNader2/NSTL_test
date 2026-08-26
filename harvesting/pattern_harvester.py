"""
harvesting/pattern_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Indexes domain-agnostic language patterns, algorithmic primitives, and control flow nodes.
"""

from __future__ import annotations
import json
import sqlite3
import os
from typing import List, Dict, Any

CORE_CODE_PATTERNS: List[Dict[str, Any]] = [
    # 1. Functional Definition & Operator Primitives
    {
        "cell_id": "PYTHON_LAMBDA_BINARY_OP",
        "domain_name": "python_core",
        "node_type": "function",
        "node_role": "function",
        "stage": 2,
        "keywords": ["function", "def", "lambda", "add", "sum", "operator", "binary", "math"],
        "inputs": {
            "a": {"type_name": "numeric", "state": "any", "required": True},
            "b": {"type_name": "numeric", "state": "any", "required": True}
        },
        "outputs": {
            "output_data": {"type_name": "numeric", "state": "computed"}
        },
        "dependencies": [],
        "code_template": "{output_var} = {a} + {b}"
    },
    {
        "cell_id": "PYTHON_PRINT_STDOUT",
        "domain_name": "python_core",
        "node_type": "function",
        "node_role": "function",
        "stage": 3,
        "keywords": ["print", "stdout", "display", "output", "show", "summary"],
        "inputs": {
            "data": {"type_name": "any", "state": "any", "required": True}
        },
        "outputs": {
            "output_data": {"type_name": "None", "state": "displayed"}
        },
        "dependencies": [],
        "code_template": "print({data})"
    },

    # 2. Data Engineering Primitives
    {
        "cell_id": "PANDAS_READ_CSV",
        "domain_name": "pandas",
        "node_type": "function",
        "node_role": "function",
        "stage": 1,
        "keywords": ["read_csv", "csv", "load", "dataframe", "ingest", "table"],
        "inputs": {
            "filepath": {"type_name": "str", "state": "source_identifier", "required": True}
        },
        "outputs": {
            "output_data": {"type_name": "DataFrame", "state": "raw"}
        },
        "dependencies": ["import pandas as pd"],
        "code_template": "{output_var} = pd.read_csv({filepath})"
    },
    {
        "cell_id": "PANDAS_DROPNA",
        "domain_name": "pandas",
        "node_type": "function",
        "node_role": "function",
        "stage": 2,
        "keywords": ["dropna", "clean", "null", "missing", "na", "filter"],
        "inputs": {
            "df": {"type_name": "DataFrame", "state": "any", "required": True}
        },
        "outputs": {
            "output_data": {"type_name": "DataFrame", "state": "cleaned"}
        },
        "dependencies": ["import pandas as pd"],
        "code_template": "{output_var} = {df}.dropna()"
    },
    {
        "cell_id": "PANDAS_SORT_VALUES",
        "domain_name": "pandas",
        "node_type": "function",
        "node_role": "function",
        "stage": 2,
        "keywords": ["sort", "sort_values", "order", "descending", "ascending", "by"],
        "inputs": {
            "df": {"type_name": "DataFrame", "state": "any", "required": True},
            "by": {"type_name": "str", "state": "column_name", "required": False, "default_value": "0"},
            "ascending": {"type_name": "bool", "state": "sort_flag", "required": False, "default_value": "True"}
        },
        "outputs": {
            "output_data": {"type_name": "DataFrame", "state": "sorted"}
        },
        "dependencies": ["import pandas as pd"],
        "code_template": "{output_var} = {df}.sort_values(by={by}, ascending={ascending})"
    },
    {
        "cell_id": "PANDAS_DESCRIBE",
        "domain_name": "pandas",
        "node_type": "function",
        "node_role": "function",
        "stage": 2,
        "keywords": ["describe", "summary", "stats", "statistics", "exploratory", "eda", "clean", "process", "transform", "values"],
        "inputs": {
            "df": {"type_name": "DataFrame", "state": "any", "required": True}
        },
        "outputs": {
            "output_data": {"type_name": "DataFrame", "state": "summary"}
        },
        "dependencies": ["import pandas as pd"],
        "code_template": "{output_var} = {df}.describe()\nprint({output_var})"
    },
    {
        "cell_id": "PANDAS_TO_CSV",
        "domain_name": "pandas",
        "node_type": "function",
        "node_role": "function",
        "stage": 3,
        "keywords": ["to_csv", "save", "export", "write", "csv"],
        "inputs": {
            "df": {"type_name": "DataFrame", "state": "any", "required": True},
            "dest_path": {"type_name": "str", "state": "dest_identifier", "required": True}
        },
        "outputs": {
            "output_data": {"type_name": "str", "state": "filepath_written"}
        },
        "dependencies": ["import pandas as pd"],
        "code_template": "{df}.to_csv({dest_path}, index=False)\n{output_var} = {dest_path}"
    },

    # 3. Image Processing Primitives (OpenCV)
    {
        "cell_id": "CV2_IMREAD",
        "domain_name": "opencv",
        "node_type": "function",
        "node_role": "function",
        "stage": 1,
        "keywords": ["imread", "read", "load", "image", "cv2", "picture"],
        "inputs": {
            "filepath": {"type_name": "str", "state": "source_identifier", "required": True}
        },
        "outputs": {
            "output_data": {"type_name": "Mat", "state": "raw"}
        },
        "dependencies": ["import cv2"],
        "code_template": "{output_var} = cv2.imread({filepath})"
    },
    {
        "cell_id": "CV2_CVTCOLOR",
        "domain_name": "opencv",
        "node_type": "function",
        "node_role": "function",
        "stage": 2,
        "keywords": ["cvtcolor", "convert", "color", "gray", "grayscale", "rgb", "hsv", "cv2"],
        "inputs": {
            "src": {"type_name": "Mat", "state": "any", "required": True},
            "code": {"type_name": "int", "state": "color_code", "required": False, "default_value": "cv2.COLOR_BGR2GRAY"}
        },
        "outputs": {
            "output_data": {"type_name": "Mat", "state": "converted"}
        },
        "dependencies": ["import cv2"],
        "code_template": "{output_var} = cv2.cvtColor({src}, {code})"
    },
    {
        "cell_id": "CV2_IMWRITE",
        "domain_name": "opencv",
        "node_type": "function",
        "node_role": "function",
        "stage": 3,
        "keywords": ["imwrite", "save", "write", "image", "export", "cv2", "output"],
        "inputs": {
            "dest_path": {"type_name": "str", "state": "dest_identifier", "required": True},
            "img": {"type_name": "Mat", "state": "any", "required": True}
        },
        "outputs": {
            "output_data": {"type_name": "bool", "state": "written"}
        },
        "dependencies": ["import cv2"],
        "code_template": "cv2.imwrite({dest_path}, {img})\n{output_var} = True"
    },

    # 4. Graph & Pathfinding Primitives
    {
        "cell_id": "PYTHON_DIJKSTRA_ALGORITHM",
        "domain_name": "algorithms",
        "node_type": "function",
        "node_role": "function",
        "stage": 2,
        "keywords": ["dijkstra", "shortest_path", "graph", "heapq", "pathfinding", "distances"],
        "inputs": {
            "graph": {"type_name": "dict", "state": "adjacency_dict", "required": False},
            "start": {"type_name": "str", "state": "source_node", "required": False}
        },
        "outputs": {
            "output_data": {"type_name": "dict", "state": "distances"}
        },
        "dependencies": ["import heapq"],
        "code_template": (
            "import heapq\n\n"
            "def dijkstra(graph, start):\n"
            "    distances = {node: float('inf') for node in graph}\n"
            "    distances[start] = 0\n"
            "    pq = [(0, start)]\n"
            "    while pq:\n"
            "        curr_dist, curr_node = heapq.heappop(pq)\n"
            "        if curr_dist > distances[curr_node]:\n"
            "            continue\n"
            "        for neighbor, weight in graph.get(curr_node, {}).items():\n"
            "            dist = curr_dist + weight\n"
            "            if dist < distances[neighbor]:\n"
            "                distances[neighbor] = dist\n"
            "                heapq.heappush(pq, (dist, neighbor))\n"
            "    return distances\n\n"
            "{output_var} = dijkstra({graph}, {start})"
        )
    }
]


def harvest_core_patterns(db_path: str):
    """Compiles clean, multi-port core patterns into the SQLite lattice."""
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for p in CORE_CODE_PATTERNS:
        cid = p["cell_id"]
        domain = p["domain_name"]
        role = p.get("node_role", "function")
        stage = p["stage"]
        keywords = json.dumps(p["keywords"])

        first_in = next(iter(p["inputs"].values()))
        first_out = next(iter(p["outputs"].values()))

        in_type = first_in["type_name"]
        in_state = first_in.get("state", "any")
        out_type = first_out["type_name"]
        out_state = first_out.get("state", "computed")

        code = p["code_template"]
        deps = json.dumps(p["dependencies"])
        schema = json.dumps(p["inputs"])

        cursor.execute("""
            INSERT OR REPLACE INTO nodes (
                cell_id, domain_name, node_type, node_role, stage, keywords,
                input_type, input_state, output_type, output_state,
                code, dependencies, configuration_schema, verified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (cid, domain, "function", role, stage, keywords, in_type, in_state, out_type, out_state, code, deps, schema))

    conn.commit()
    conn.close()
    print(f"[+] Successfully harvested {len(CORE_CODE_PATTERNS)} core patterns into {db_path}")


if __name__ == "__main__":
    db = os.path.join(os.path.dirname(__file__), "..", "trees", "lattice.db")
    harvest_core_patterns(db)
