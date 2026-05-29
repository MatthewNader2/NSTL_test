# lattice.py


class MacroNode:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.micro_cells = {}

    def register_cell(self, cell_id: str, cell_instance):
        self.micro_cells[cell_id] = cell_instance


class MicroCell:
    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        code_template: str,
        inputs: dict,
        outputs: dict,
    ):
        self.cell_id = cell_id
        self.stage = (
            stage  # 1: Ingestion, 2: Cleaning, 3: Processing, 4: Terminal Action
        )
        self.keywords = keywords  # Latent semantic anchor words for token matching
        self.code_template = code_template
        self.inputs = inputs
        self.outputs = outputs


# ==========================================
# EXTENDED TOPOLOGY WITH SEMANTIC METADATA
# ==========================================

file_ops = MacroNode("File_Operations", "Handling local I/O and ingestion")
data_ops = MacroNode("Data_Transformations", "Pandas manipulations and mutations")
algo_ds = MacroNode(
    "Algorithms_and_DS", "Classic computer science sorting and searching structures"
)

# Cell 1: Ingestion Stage (Stage 1)
cell_read_csv = MicroCell(
    cell_id="READ_CSV",
    stage=1,
    keywords={"read", "csv", "load", "import", "ingest", "file", "database"},
    code_template="with open('{input_var}', 'r') as f:\n    {output_var} = pd.read_csv(f)",
    inputs={"input_type": "str", "expected_state": "filepath"},
    outputs={"output_type": "DataFrame", "resulting_state": "raw_data"},
)

# Cell 2: Cleaning Stage (Stage 2)
cell_filter_nan = MicroCell(
    cell_id="FILTER_NAN",
    stage=2,
    keywords={"clean", "nan", "drop", "strip", "null", "missing", "filter"},
    code_template="{output_var} = {input_var}.dropna()",
    inputs={"input_type": "DataFrame", "expected_state": "raw_data"},
    outputs={"output_type": "DataFrame", "resulting_state": "clean_data"},
)

# Cell 3a: Sorting DataFrame (Stage 3)
cell_sort_dataframe = MicroCell(
    cell_id="SORT_DF",
    stage=3,
    keywords={"sort", "order", "arrange", "sequence", "ascending", "securely"},
    code_template="{output_var} = {input_var}.sort_values(by={input_var}.columns[0], ascending=True)",
    inputs={"input_type": "DataFrame", "expected_state": "clean_data"},
    outputs={"output_type": "DataFrame", "resulting_state": "sorted_data"},
)

# Cell 3b: Sorting List Primitive (Stage 3)
cell_sort_primitive_list = MicroCell(
    cell_id="SORT_LIST",
    stage=3,
    keywords={"sort", "order", "arrange", "sequence", "organize", "messy"},
    code_template="{output_var} = sorted({input_var})",
    inputs={"input_type": "list", "expected_state": "unordered_collection"},
    outputs={"output_type": "list", "resulting_state": "sorted_collection"},
)

# Cell 4: Searching Stage (Stage 4)
cell_binary_search = MicroCell(
    cell_id="BINARY_SEARCH",
    stage=4,
    keywords={"find", "search", "location", "binary", "item", "index"},
    code_template=(
        "def binary_search(arr, target):\n"
        "    low, high = 0, len(arr) - 1\n"
        "    while low <= high:\n"
        "        mid = (low + high) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: low = mid + 1\n"
        "        else: high = mid - 1\n"
        "    return -1\n"
        "{output_var} = binary_search({input_var}, target_value)"
    ),
    inputs={"input_type": "list", "expected_state": "sorted_collection"},
    outputs={"output_type": "int", "resulting_state": "target_index"},
)

file_ops.register_cell("READ_CSV", cell_read_csv)
data_ops.register_cell("FILTER_NAN", cell_filter_nan)
data_ops.register_cell("SORT_DF", cell_sort_dataframe)
algo_ds.register_cell("SORT_LIST", cell_sort_primitive_list)
algo_ds.register_cell("BINARY_SEARCH", cell_binary_search)

GLOBAL_LATTICE = {
    "File_Operations": file_ops,
    "Data_Transformations": data_ops,
    "Algorithms_and_DS": algo_ds,
}
