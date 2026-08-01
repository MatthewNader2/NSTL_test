import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any
from abc import ABC
from log_config import get_logger

# Module-level lock for synthesized_nodes.json file operations
_synth_file_lock = threading.Lock()
logger = get_logger('lattice')

@dataclass(slots=True)
class AlgebraicSignature:
    type_name: str
    state: str

    def matches(self, other: 'AlgebraicSignature') -> bool:
        """Structural unification with 'any' wildcard support."""
        if self.type_name != "any" and other.type_name != "any":
            if self.type_name != other.type_name:
                return False
        if self.state != "any" and other.state != "any":
            if self.state != other.state:
                return False
        return True

class Cell(ABC):
    __slots__ = ["cell_id", "stage", "keywords", "type", "inputs", "outputs", "domain_name", "node_type", "_db_path"]

    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        cell_type: str,
        inputs: AlgebraicSignature,
        outputs: AlgebraicSignature,
        domain_name: str = "",
        node_type: str = "function",
        db_path: str = ""
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(keywords) if keywords else set()
        self.type = cell_type
        self.inputs = inputs
        self.outputs = outputs
        self.domain_name = domain_name
        self.node_type = node_type
        self._db_path = db_path

class MicroCell(Cell):
    __slots__ = ["_code_template", "intent_expansion", "matched_heuristics"]
    # Class-level connection cache: avoids opening 100+ connections during unrolling
    _conn_cache: dict = {}

    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        inputs: AlgebraicSignature,
        outputs: AlgebraicSignature,
        domain_name: str = "",
        node_type: str = "function",
        cell_type: str = "micro",
        intent_expansion: list = None,
        db_path: str = ""
    ):
        super().__init__(cell_id, stage, keywords, cell_type, inputs, outputs, domain_name, node_type, db_path)
        self._code_template = None
        self.intent_expansion = intent_expansion or []
        self.matched_heuristics = []

    @classmethod
    def _get_cached_conn(cls, db_path: str):
        """Returns a cached SQLite connection for the given path."""
        if db_path not in cls._conn_cache:
            cls._conn_cache[db_path] = sqlite3.connect(db_path, check_same_thread=False)
        return cls._conn_cache[db_path]

    @property
    def code_template(self) -> str:
        if self._code_template is not None:
            return self._code_template
        # Lazy load from SQLite to save massive amounts of RAM
        if not self._db_path or not os.path.exists(self._db_path):
            return ""
        try:
            conn = self._get_cached_conn(self._db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT code FROM nodes WHERE cell_id = ?", (self.cell_id,))
            row = cursor.fetchone()
            if row:
                self._code_template = row[0]
                return self._code_template
        except Exception as e:
            logger.error(f"[SQLite] Error fetching code template for {self.cell_id}: {e}")
        return ""
    
    @code_template.setter
    def code_template(self, value):
        self._code_template = value

class MacroCell(Cell):
    __slots__ = ["algorithmic_steps"]

    def __init__(
        self,
        cell_id: str,
        stage: int,
        keywords: set,
        inputs: AlgebraicSignature,
        outputs: AlgebraicSignature,
        domain_name: str = "",
        node_type: str = "function",
        cell_type: str = "macro",
        algorithmic_steps: List[str] = None,
        db_path: str = ""
    ):
        super().__init__(cell_id, stage, keywords, cell_type, inputs, outputs, domain_name, node_type, db_path)
        self.algorithmic_steps = algorithmic_steps or []

class LatticeOrchestrator:
    def __init__(self, trees_directory="trees", active_domain="Python_Core"):
        self.trees_directory = trees_directory
        self.db_path = os.path.join(trees_directory, "lattice.db")
        self.active_domain = active_domain
        self.loaded_cells: Dict[str, Cell] = {}
        self.topology: Dict[str, List[str]] = {}
        # Persistent reverse-lookup: (input_type, input_state) -> list[Cell]
        self._cells_by_input: Dict[tuple, List[Cell]] = {}
        self.load_from_database()
        self.build_topology()

    def _parse_signature(self, raw_dict: dict, type_key: str, state_key: str) -> AlgebraicSignature:
        if not raw_dict:
            return AlgebraicSignature("", "")
        type_name = raw_dict.get(type_key, raw_dict.get("type_name", ""))
        state = raw_dict.get(state_key, raw_dict.get("state", ""))
        return AlgebraicSignature(type_name=type_name, state=state)

    def _parse_cell(self, raw_cell: dict) -> Cell:
        inputs_sig = self._parse_signature(raw_cell.get("inputs", {}), "input_type", "expected_state")
        outputs_sig = self._parse_signature(raw_cell.get("outputs", {}), "output_type", "resulting_state")
        
        cell_type = raw_cell.get("type", "micro")
        if cell_type == "macro":
            return MacroCell(
                cell_id=raw_cell.get("cell_id", "UNKNOWN"),
                stage=raw_cell.get("stage", 0),
                keywords=raw_cell.get("keywords", []),
                inputs=inputs_sig,
                outputs=outputs_sig,
                cell_type=cell_type,
                algorithmic_steps=raw_cell.get("algorithmic_steps", [])
            )
        else:
            implementations = raw_cell.get("domain_implementations", {})
            domain_data = implementations.get(self.active_domain, {})
            active_code = domain_data.get("code", raw_cell.get("code_template", ""))
            
            mc = MicroCell(
                cell_id=raw_cell.get("cell_id", "UNKNOWN"),
                stage=raw_cell.get("stage", 0),
                keywords=raw_cell.get("keywords", []),
                inputs=inputs_sig,
                outputs=outputs_sig,
                cell_type=cell_type,
                intent_expansion=raw_cell.get("intent_expansion", []),
            )
            mc.code_template = active_code
            return mc

    def load_from_database(self):
        if not os.path.exists(self.db_path):
            logger.info(f"[LATTICE] No SQLite DB found at {self.db_path}")
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT cell_id, domain_name, node_type, stage, keywords, input_type, input_state, output_type, output_state FROM nodes")
            rows = cursor.fetchall()
            
            for row in rows:
                cell_id, domain_name, node_type, stage, keywords_json, in_type, in_state, out_type, out_state = row
                
                try:
                    keywords = json.loads(keywords_json) if keywords_json else []
                except (json.JSONDecodeError, ValueError, TypeError):
                    keywords = []
                    
                in_sig = AlgebraicSignature(type_name=in_type, state=in_state)
                out_sig = AlgebraicSignature(type_name=out_type, state=out_state)
                
                # Assume all from SQLite are micro cells currently
                cell = MicroCell(
                    cell_id=cell_id,
                    stage=stage,
                    keywords=set(keywords),
                    inputs=in_sig,
                    outputs=out_sig,
                    domain_name=domain_name,
                    node_type=node_type,
                    db_path=self.db_path
                )
                self.loaded_cells[cell.cell_id] = cell
                logger.debug(f"Loaded cell {cell.cell_id} from SQLite.")
                
            conn.close()
            logger.info(f"[LATTICE] Successfully loaded {len(self.loaded_cells)} nodes from SQLite.")
        except Exception as e:
            logger.error(f"[LATTICE] Error loading from SQLite: {e}")

    def inject_transient_macro(self, macro_dict: dict) -> Cell:
        """Injects LLM-generated Macro-Nodes dynamically.
        Uses incremental O(N) topology update instead of full O(N²) rebuild."""
        cell = self._parse_cell(macro_dict)
        self.loaded_cells[cell.cell_id] = cell
        self._add_cell_to_topology(cell)
        
        # Persist the synthesized node dynamically to disk (thread-safe)
        synth_file = os.path.join(self.trees_directory, "micro", "synthesized_nodes.json")
        with _synth_file_lock:
            try:
                if os.path.exists(synth_file):
                    with open(synth_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = {"domain_name": "Synthesized_Domain", "cells": []}
                    
                data.setdefault("cells", []).append(macro_dict)
                with open(synth_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"[LATTICE WARNING] Failed to persist synthesized node: {e}")
            
        return cell

    def _get_all_cells_recursively(self, cells: List[Cell]) -> List[Cell]:
        return cells

    def build_topology(self):
        """Full O(N) topology build. Called once at startup."""
        self.topology = {}
        self._cells_by_input = {}
        all_cells = self._get_all_cells_recursively(list(self.loaded_cells.values()))

        # Pass 1 — initialise adjacency lists and build reverse lookup O(N)
        for cell in all_cells:
            self.topology[cell.cell_id] = []
            sig = (cell.inputs.type_name, cell.inputs.state)
            self._cells_by_input.setdefault(sig, []).append(cell)

        # Pass 2 — wire edges O(N) using the reverse lookup
        for cell_a in all_cells:
            out_sig = (cell_a.outputs.type_name, cell_a.outputs.state)
            for cell_b in self._cells_by_input.get(out_sig, []):
                if cell_b.cell_id not in self.topology[cell_a.cell_id]:
                    self.topology[cell_a.cell_id].append(cell_b.cell_id)

        edge_count = sum(len(edges) for edges in self.topology.values())
        logger.info(f"Built topology with {edge_count} edges.")

    def _add_cell_to_topology(self, cell: Cell):
        """Incremental O(N) topology update for a single newly injected cell.
        Avoids the full O(N²) rebuild that build_topology() would trigger."""
        logger.debug(f"Adding cell {cell.cell_id} to topology incrementally.")
        # Register the new cell's adjacency entry
        self.topology.setdefault(cell.cell_id, [])

        # Update reverse lookup
        in_sig = (cell.inputs.type_name, cell.inputs.state)
        bucket = self._cells_by_input.setdefault(in_sig, [])
        if not any(c.cell_id == cell.cell_id for c in bucket):
            bucket.append(cell)

        # Wire: who can flow INTO the new cell? (existing cells whose output matches new cell's input)
        for existing in self.loaded_cells.values():
            if existing.cell_id == cell.cell_id:
                continue
            out_sig = (existing.outputs.type_name, existing.outputs.state)
            if out_sig == in_sig:
                existing_edges = self.topology.setdefault(existing.cell_id, [])
                if cell.cell_id not in existing_edges:
                    existing_edges.append(cell.cell_id)

        # Wire: where can the new cell flow TO? (use reverse lookup — O(1) amortised)
        out_sig = (cell.outputs.type_name, cell.outputs.state)
        for target in self._cells_by_input.get(out_sig, []):
            if target.cell_id != cell.cell_id and target.cell_id not in self.topology[cell.cell_id]:
                self.topology[cell.cell_id].append(target.cell_id)

    def get_all_available_cells(self) -> list:
        return self._get_all_cells_recursively(list(self.loaded_cells.values()))

    def get_neighbors(self, cell_id: str) -> list:
        neighbor_ids = self.topology.get(cell_id, [])
        return [self.loaded_cells[nid] for nid in neighbor_ids if nid in self.loaded_cells]

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
