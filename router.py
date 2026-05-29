# router.py
from lattice import GLOBAL_LATTICE


class LatticeRouter:
    def __init__(self):
        self.available_cells = {}
        for neighborhood in GLOBAL_LATTICE.values():
            for cell_id, cell in neighborhood.micro_cells.items():
                self.available_cells[cell_id] = cell

    def plan_path(self, user_intent: str) -> list:
        path = []
        intent_lower = user_intent.lower()

        # Ingestion rule
        if "csv" in intent_lower or "read" in intent_lower:
            path.append(self.available_cells["READ_CSV"])

        # Cleaning rule
        if "clean" in intent_lower or "nan" in intent_lower:
            path.append(self.available_cells["FILTER_NAN"])

        # Polymorphic sorting routing intent
        if "sort" in intent_lower or "order" in intent_lower:
            # Note: The router passes BOTH candidate cells.
            # The Unification Gate filters the valid one based on type match.
            if "SORT_DF" in self.available_cells:
                path.append(self.available_cells["SORT_DF"])
            if "SORT_LIST" in self.available_cells:
                path.append(self.available_cells["SORT_LIST"])

        # Finding/Search rule
        if "find" in intent_lower or "search" in intent_lower:
            path.append(self.available_cells["BINARY_SEARCH"])

        return path
