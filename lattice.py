# lattice.py

class MacroNode:
    """Represents a high-level logical neighborhood in the lattice."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.micro_cells = {}

    def register_cell(self, cell_id: str, cell_instance):
        self.micro_cells[cell_id] = cell_instance


class MicroCell:
    """
    Represents an atomic best-practice logic primitive.
    Equivalent to a safe AST snippet.
    """
    def __init__(self, cell_id: str, code_template: str, inputs: dict, outputs: dict):
        self.cell_id = cell_id
        # The template uses placeholders for dynamic variable binding
        self.code_template = code_template
        # Type and state signature constraints
        self.inputs = inputs   # e.g., {"type": "str", "state": "path_unverified"}
        self.outputs = outputs # e.g., {"type": "DataFrame", "state": "dirty"}


# ==========================================
# INSTANTIATING A TOY LATTICE FOR PYTHON
# ==========================================

# 1. Initialize Macro Neighborhoods
file_ops = MacroNode("File_Operations", "Handling local I/O and ingestion")
data_ops = MacroNode("Data_Transformations", "Parsing, filtering, and mutating in-memory structures")

# 2. Define Best-Practice Micro Cells (Snippets)

# Cell A: Safe File Reading
cell_read_csv = MicroCell(
    cell_id="READ_CSV",
    code_template="with open('{input_var}', 'r') as f:\n    {output_var} = pd.read_csv(f)",
    inputs={"input_type": "str", "expected_state": "filepath"},
    outputs={"output_type": "DataFrame", "resulting_state": "raw_data"}
)

# Cell B: Data Filtering
cell_filter_nan = MicroCell(
    cell_id="FILTER_NAN",
    code_template="{output_var} = {input_var}.dropna()",
    inputs={"input_type": "DataFrame", "expected_state": "raw_data"},
    outputs={"output_type": "DataFrame", "resulting_state": "clean_data"}
)

# Cell C: Convert Data to JSON String
cell_to_json = MicroCell(
    cell_id="TO_JSON",
    code_template="{output_var} = {input_var}.to_json(orient='records')",
    inputs={"input_type": "DataFrame", "expected_state": "clean_data"},
    outputs={"output_type": "str", "resulting_state": "serialized_json"}
)

# 3. Register micro cells into macro neighborhoods (Building the Topology)
file_ops.register_cell("READ_CSV", cell_read_csv)
data_ops.register_cell("FILTER_NAN", cell_filter_nan)
data_ops.register_cell("TO_JSON", cell_to_json)

# Global Lattice Map for our system
GLOBAL_LATTICE = {
    "File_Operations": file_ops,
    "Data_Transformations": data_ops
}
