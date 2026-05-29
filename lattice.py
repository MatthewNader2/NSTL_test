# lattice.py


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
        self.keywords = keywords
        self.code_template = code_template
        self.inputs = inputs
        self.outputs = outputs


class DomainLattice:
    """Represents an isolated, independent Domain Tree (e.g., Pandas library or a Robotics HAL)."""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.cells = {}

    def add_cell(self, cell: MicroCell):
        self.cells[cell.cell_id] = cell


class LatticeOrchestrator:
    """
    The 'Wrapper'. Inspects all active domain trees and
    informs the router of global capabilities.
    """

    def __init__(self):
        self.registered_domains = {}

    def register_domain_tree(self, domain_tree: DomainLattice):
        """Plugs an entire functional tree into the runtime engine."""
        self.registered_domains[domain_tree.domain_name] = domain_tree
        print(
            f"[ORCHESTRATOR] Successfully mounted domain tree: '{domain_tree.domain_name}'"
        )

    def unmount_domain_tree(self, domain_name: str):
        """Removes a domain tree, dynamically stripping capabilities from the AI."""
        if domain_name in self.registered_domains:
            del self.registered_domains[domain_name]
            print(f"[ORCHESTRATOR] Unmounted domain tree: '{domain_name}'")

    def get_all_available_cells(self) -> list:
        """Collects every cell from all active trees to hand over to the router."""
        all_cells = []
        for domain_tree in self.registered_domains.values():
            for cell in domain_tree.cells.values():
                all_cells.append(cell)
        return all_cells
