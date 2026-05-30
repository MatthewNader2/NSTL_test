# lattice.py
import json
import os


class MicroCell:
    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        code_template: str,
        inputs: dict,
        outputs: dict,
        cell_type: str = "micro",
        intent_expansion: list = None,
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(keywords)
        self.type = cell_type
        self.intent_expansion = intent_expansion or []
        self.code_template = code_template
        self.inputs = inputs or {}
        self.outputs = outputs or {}


class LatticeOrchestrator:
    def __init__(self, trees_directory="trees"):
        self.trees_directory = trees_directory
        self.loaded_cells = {}
        self.topology = {}
        self.discover_and_load_trees()
        self.build_topology()

    def discover_and_load_trees(self):
        if not os.path.exists(self.trees_directory):
            return

        for file_name in os.listdir(self.trees_directory):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.trees_directory, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        tree_data = json.load(f)

                    for raw_cell in tree_data.get("cells", []):
                        cell = MicroCell(
                            cell_id=raw_cell.get("cell_id", "UNKNOWN"),
                            stage=raw_cell.get("stage", 0),
                            keywords=raw_cell.get("keywords", []),
                            code_template=raw_cell.get("code_template", ""),
                            inputs=raw_cell.get("inputs", {}),
                            outputs=raw_cell.get("outputs", {}),
                            cell_type=raw_cell.get("type", "micro"),
                            intent_expansion=raw_cell.get("intent_expansion", []),
                        )
                        self.loaded_cells[cell.cell_id] = cell
                except Exception as e:
                    pass

    def build_topology(self):
        for cell_id in self.loaded_cells:
            self.topology[cell_id] = []

        for cell_a in self.loaded_cells.values():
            if cell_a.type == "macro":
                continue
            for cell_b in self.loaded_cells.values():
                if cell_b.type == "macro":
                    continue
                if cell_a.outputs.get("output_type") == cell_b.inputs.get(
                    "input_type"
                ) and cell_a.outputs.get("resulting_state") == cell_b.inputs.get(
                    "expected_state"
                ):
                    self.topology[cell_a.cell_id].append(cell_b.cell_id)

    def get_all_available_cells(self) -> list:
        return list(self.loaded_cells.values())

    def get_neighbors(self, cell_id: str) -> list:
        neighbor_ids = self.topology.get(cell_id, [])
        return [self.loaded_cells[nid] for nid in neighbor_ids]

    # RESTORED AND FIXED:
    def find_type_bridge(self, from_type: str, to_type: str):
        """Searches the entire lattice for a micro-node that can cast from one type to another."""
        for cell in self.loaded_cells.values():
            if (
                cell.type == "micro"
                and cell.inputs.get("input_type") == from_type
                and cell.outputs.get("output_type") == to_type
            ):
                return cell
        return None
