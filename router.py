# router.py
from lattice import GLOBAL_LATTICE


class LatticeRouter:
    """
    Mocks the policy network. Analyzes user intent strings and
    determines the optimal traversal path coordinates across the lattice.
    """

    def __init__(self):
        # Flatten our hierarchical graph layout into a searchable index for the router
        self.available_cells = {}
        for neighborhood in GLOBAL_LATTICE.values():
            for cell_id, cell in neighborhood.micro_cells.items():
                self.available_cells[cell_id] = cell

    def plan_path(self, user_intent: str) -> list:
        """
        Calculates the execution path.
        In a full build, this performs real-time graph topological sorting or state space pathfinding.
        """
        path = []
        intent_lower = user_intent.lower()

        # Step 1: Infer ingestion requirements
        if "csv" in intent_lower or "read" in intent_lower:
            if "READ_CSV" in self.available_cells:
                path.append(self.available_cells["READ_CSV"])

        # Step 2: Infer processing/transformation steps
        if "clean" in intent_lower or "filter" in intent_lower or "nan" in intent_lower:
            if "FILTER_NAN" in self.available_cells:
                path.append(self.available_cells["FILTER_NAN"])

        # Step 3: Infer serialization/output structures
        if "json" in intent_lower or "serialize" in intent_lower:
            if "TO_JSON" in self.available_cells:
                path.append(self.available_cells["TO_JSON"])

        print(
            f"[ROUTER PLAN GENERATED] Path sequence: {[cell.cell_id for cell in path]}"
        )
        return path
