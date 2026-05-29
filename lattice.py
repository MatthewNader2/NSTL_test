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
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(keywords)  # Convert JSON array to searchable Python set
        self.code_template = code_template
        self.inputs = inputs
        self.outputs = outputs


class LatticeOrchestrator:
    """
    The File-Based Ontology Wrapper. Automatically discovers and parses
    independent domain tree files from the local storage disk at runtime.
    """

    def __init__(self, trees_directory="trees"):
        self.trees_directory = trees_directory
        self.loaded_cells = {}
        # Execute auto-discovery sweep instantly upon runtime boot
        self.discover_and_load_trees()

    def discover_and_load_trees(self):
        """Scans the designated directory for valid standalone domain tree JSON files."""
        if not os.path.exists(self.trees_directory):
            print(
                f"[ORCHESTRATOR ERROR] Target directory '{self.trees_directory}' does not exist."
            )
            return

        print("[ORCHESTRATOR] Initiating automated file-system discovery pass...")

        for file_name in os.listdir(self.trees_directory):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.trees_directory, file_name)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        tree_data = json.load(f)

                    domain = tree_data.get("domain_name", "Unknown_Domain")
                    cell_count = 0

                    # Parse each raw JSON block directly into memory engines
                    for raw_cell in tree_data.get("cells", []):
                        cell = MicroCell(
                            cell_id=raw_cell["cell_id"],
                            stage=raw_cell["stage"],
                            keywords=raw_cell["keywords"],
                            code_template=raw_cell["code_template"],
                            inputs=raw_cell["inputs"],
                            outputs=raw_cell["outputs"],
                        )
                        # Register uniquely under global capabilities tracking index
                        self.loaded_cells[cell.cell_id] = cell
                        cell_count += 1

                    print(
                        f" -> Found and successfully mounted module: '{domain}' ({cell_count} cells loaded)"
                    )

                except Exception as e:
                    print(
                        f"[ORCHESTRATOR ERROR] Failed to safely parse tree file {file_name}: {str(e)}"
                    )

    def get_all_available_cells(self) -> list:
        """Exposes discovered cell objects directly to the routing network loop."""
        return list(self.loaded_cells.values())
