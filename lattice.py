# lattice.py
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any
from abc import ABC

@dataclass(slots=True)
class AlgebraicSignature:
    type_name: str
    state: str

    def matches(self, other: 'AlgebraicSignature') -> bool:
        """Strict mathematical type validation."""
        return self.type_name == other.type_name and self.state == other.state

class Cell(ABC):
    __slots__ = ["cell_id", "stage", "keywords", "type", "inputs", "outputs"]

    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        cell_type: str,
        inputs: AlgebraicSignature,
        outputs: AlgebraicSignature,
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(keywords) if keywords else set()
        self.type = cell_type
        self.inputs = inputs
        self.outputs = outputs

class MicroCell(Cell):
    __slots__ = ["code_template", "intent_expansion"]

    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        code_template: str,
        inputs: AlgebraicSignature,
        outputs: AlgebraicSignature,
        cell_type: str = "micro",
        intent_expansion: list = None,
    ):
        super().__init__(cell_id, stage, keywords, cell_type, inputs, outputs)
        self.code_template = code_template
        self.intent_expansion = intent_expansion or []

class MacroCell(Cell):
    __slots__ = ["sub_cells", "internal_topology"]

    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        inputs: AlgebraicSignature,
        outputs: AlgebraicSignature,
        cell_type: str = "macro",
        sub_cells: List['Cell'] = None,
        internal_topology: Dict[str, List[str]] = None,
    ):
        super().__init__(cell_id, stage, keywords, cell_type, inputs, outputs)
        self.sub_cells = sub_cells or []
        self.internal_topology = internal_topology or {}

class LatticeOrchestrator:
    def __init__(self, trees_directory="trees", active_domain="Python_Core"):
        self.trees_directory = trees_directory
        self.active_domain = active_domain
        self.loaded_cells: Dict[str, Cell] = {}
        self.topology: Dict[str, List[str]] = {}
        self.discover_and_load_trees()
        self.build_topology()

    def _parse_signature(self, raw_dict: dict, type_key: str, state_key: str) -> AlgebraicSignature:
        if not raw_dict:
            return AlgebraicSignature("", "")
        # BUG 3 FIX: Support both old-schema keys (input_type/expected_state/output_type/resulting_state)
        # AND new-schema keys (type_name/state) used by planner.py and synthesis.py LLM output.
        # Try the specific legacy key first, fall back to the new unified key name.
        type_name = raw_dict.get(type_key, raw_dict.get("type_name", ""))
        state = raw_dict.get(state_key, raw_dict.get("state", ""))
        return AlgebraicSignature(type_name=type_name, state=state)

    def _parse_cell(self, raw_cell: dict) -> Cell:
        inputs_sig = self._parse_signature(raw_cell.get("inputs", {}), "input_type", "expected_state")
        outputs_sig = self._parse_signature(raw_cell.get("outputs", {}), "output_type", "resulting_state")
        
        cell_type = raw_cell.get("type", "micro")
        if cell_type == "macro":
            # TEMPORARILY store raw sub_cell IDs, resolved in Pass 2
            sub_cells = raw_cell.get("sub_cells", [])
            return MacroCell(
                cell_id=raw_cell.get("cell_id", "UNKNOWN"),
                stage=raw_cell.get("stage", 0),
                keywords=raw_cell.get("keywords", []),
                inputs=inputs_sig,
                outputs=outputs_sig,
                cell_type=cell_type,
                sub_cells=sub_cells,
                internal_topology=raw_cell.get("internal_topology", {})
            )
        else:
            # R↓ Optimization: Only load the domain code we need
            implementations = raw_cell.get("domain_implementations", {})
            domain_data = implementations.get(self.active_domain, {})
            # Fallback to old code_template if domain_implementations is missing
            active_code = domain_data.get("code", raw_cell.get("code_template", ""))
            
            return MicroCell(
                cell_id=raw_cell.get("cell_id", "UNKNOWN"),
                stage=raw_cell.get("stage", 0),
                keywords=raw_cell.get("keywords", []),
                code_template=active_code,
                inputs=inputs_sig,
                outputs=outputs_sig,
                cell_type=cell_type,
                intent_expansion=raw_cell.get("intent_expansion", []),
            )

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
                        cell = self._parse_cell(raw_cell)
                        self.loaded_cells[cell.cell_id] = cell
                except Exception as e:
                    # BUG 19 FIX: Log errors instead of silently swallowing them.
                    print(f"[LATTICE WARNING] Failed to load '{file_name}': {e}")

        # PASS 2: Link Macro sub_cells
        for cell in self.loaded_cells.values():
            if isinstance(cell, MacroCell):
                resolved_sub_cells = []
                for sub_id in cell.sub_cells:
                    if isinstance(sub_id, str) and sub_id in self.loaded_cells:
                        resolved_sub_cells.append(self.loaded_cells[sub_id])
                    elif isinstance(sub_id, Cell): # Fallback
                        resolved_sub_cells.append(sub_id)
                cell.sub_cells = resolved_sub_cells

    def inject_transient_macro(self, macro_dict: dict) -> Cell:
        """Injects LLM-generated Macro-Nodes dynamically."""
        cell = self._parse_cell(macro_dict)
        self.loaded_cells[cell.cell_id] = cell
        
        # Pass 2 resolution for this specific transient node
        if isinstance(cell, MacroCell):
            resolved_sub_cells = []
            for sub_id in cell.sub_cells:
                if isinstance(sub_id, str) and sub_id in self.loaded_cells:
                    resolved_sub_cells.append(self.loaded_cells[sub_id])
                elif isinstance(sub_id, Cell):
                    resolved_sub_cells.append(sub_id)
            cell.sub_cells = resolved_sub_cells

        # Rebuild global topology to include the new path
        self.build_topology()
        return cell

    def _get_all_cells_recursively(self, cells: List[Cell]) -> List[Cell]:
        all_cells = []
        for cell in cells:
            all_cells.append(cell)
            if isinstance(cell, MacroCell):
                all_cells.extend(self._get_all_cells_recursively(cell.sub_cells))
        return all_cells

    def build_topology(self):
        # BUG 8 FIX: Reset topology dict entirely at the start of each build
        # to prevent duplicate edges from accumulating across repeated calls.
        self.topology = {}

        all_cells = self._get_all_cells_recursively(list(self.loaded_cells.values()))
        
        for cell in all_cells:
            self.topology[cell.cell_id] = []

        for cell in all_cells:
            if isinstance(cell, MacroCell):
                for src_id, dest_ids in cell.internal_topology.items():
                    if src_id in self.topology:
                        self.topology[src_id].extend(dest_ids)

        for cell_a in all_cells:
            for cell_b in all_cells:
                if cell_a.outputs.matches(cell_b.inputs):
                    if cell_b.cell_id not in self.topology[cell_a.cell_id]:
                        self.topology[cell_a.cell_id].append(cell_b.cell_id)

    def get_all_available_cells(self) -> list:
        return self._get_all_cells_recursively(list(self.loaded_cells.values()))

    def get_neighbors(self, cell_id: str) -> list:
        neighbor_ids = self.topology.get(cell_id, [])
        all_cells = {c.cell_id: c for c in self.get_all_available_cells()}
        return [all_cells[nid] for nid in neighbor_ids if nid in all_cells]

    # RESTORED AND FIXED:
    def find_type_bridge(self, from_type: str, to_type: str) -> Optional[Cell]:
        """Searches the entire lattice for a micro-node that can cast from one type to another."""
        for cell in self.get_all_available_cells():
            if (
                isinstance(cell, MicroCell)
                and cell.inputs.type_name == from_type
                and cell.outputs.type_name == to_type
            ):
                return cell
        return None
