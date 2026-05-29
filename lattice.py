# lattice.py


class MacroNode:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.micro_cells = {}

    def register_cell(self, cell_id: str, cell_instance):
        self.micro_cells[cell_id] = cell_instance


class MicroCell:
    def __init__(self, cell_id: str, code_template: str, inputs: dict, outputs: dict):
        self.cell_id = cell_id
        self.code_template = code_template
        self.inputs = inputs
        self.outputs = outputs


# ==========================================
# EXTENDED TOPOLOGY INDICES
# ==========================================

# 1. Initialize Macro Neighborhoods
file_ops = MacroNode("File_Operations", "Handling local I/O and ingestion")
data_ops = MacroNode("Data_Transformations", "Pandas manipulations and mutations")
algo_ds = MacroNode(
    "Algorithms_and_DS", "Classic computer science sorting and searching structures"
)

# 2. Existing Pandas Ingestion / Cleaning Cells
cell_read_csv = MicroCell(
    cell_id="READ_CSV",
    code_template="with open('{input_var}', 'r') as f:\n    {output_var} = pd.read_csv(f)",
    inputs={"input_type": "str", "expected_state": "filepath"},
    outputs={"output_type": "DataFrame", "resulting_state": "raw_data"},
)

cell_filter_nan = MicroCell(
    cell_id="FILTER_NAN",
    code_template="{output_var} = {input_var}.dropna()",
    inputs={"input_type": "DataFrame", "expected_state": "raw_data"},
    outputs={"output_type": "DataFrame", "resulting_state": "clean_data"},
)

# 3. NEW: Polymorphic Sorting Cells
cell_sort_dataframe = MicroCell(
    cell_id="SORT_DF",
    code_template="{output_var} = {input_var}.sort_values(by={input_var}.columns[0], ascending=True)",
    inputs={"input_type": "DataFrame", "expected_state": "clean_data"},
    outputs={"output_type": "DataFrame", "resulting_state": "sorted_data"},
)

cell_sort_primitive_list = MicroCell(
    cell_id="SORT_LIST",
    code_template="{output_var} = sorted({input_var})",
    inputs={"input_type": "list", "expected_state": "unordered_collection"},
    outputs={"output_type": "list", "resulting_state": "sorted_collection"},
)

# 4. NEW: Finding / Common Data Structures (Binary Search Primitive)
cell_binary_search = MicroCell(
    cell_id="BINARY_SEARCH",
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

# Registering all cells to their respective MacroNodes
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
