"""
src/lattice.py - Neuro-Symbolic Topological Lattice (NSTL)
Defines formal algebraic type signatures, multi-port cell nodes,
and ultra-lightweight on-demand topology resolution.
"""

from __future__ import annotations
import functools
import json
import os
import re
import sqlite3
import threading
from collections import deque
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, FrozenSet, Union
from log_config import get_logger

logger = get_logger('lattice')


class TypeRegistry:
    """Dynamic Poset Type Hierarchy with safe incremental registration."""
    _instance: Optional[TypeRegistry] = None
    _lock = threading.RLock()

    def __init__(self):
        self._parents: Dict[str, Set[str]] = {}
        self._aliases: Dict[str, str] = {}
        self._register_default_types()

    @property
    def poset(self) -> Dict[str, Set[str]]:
        return self._parents

    @classmethod
    def get_instance(cls) -> TypeRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _register_default_types(self):
        # Register base types first to avoid forward-reference issues
        self.register_type("any")
        self.register_type("object", super_type="any")
        self.register_type("numeric", super_type="object")
        self.register_type("None", super_type="any")

        # Register leaf types
        self.register_type("str", super_type="object")
        self.register_type("int", super_type="numeric")
        self.register_type("float", super_type="numeric")
        self.register_type("bool", super_type="int")
        self.register_type("list", super_type="object")
        self.register_type("dict", super_type="object")
        self.register_type("tuple", super_type="object")
        self.register_type("ndarray", super_type="object")
        self.register_type("Mat", super_type="ndarray")
        self.register_type("DataFrame", super_type="object")
        self.register_type("Series", super_type="object")
        self.register_type("Figure", super_type="object")
        self.register_type("Graph", super_type="object")

    def register_type(self, type_name: str, super_type: Optional[str] = None):
        name = type_name.strip()
        if not name:
            return
        if name not in self._parents:
            self._parents[name] = set()
        if super_type:
            super_name = super_type.strip()
            if super_name and super_name not in self._parents:
                self._parents[super_name] = set()
            self._parents[name].add(super_name)
        try:
            self.is_subtype.cache_clear()
        except AttributeError:
            pass

    def canonical_name(self, type_name: str) -> str:
        if not type_name:
            return "any"
        clean = str(type_name).strip()
        return self._aliases.get(clean.lower(), clean)

    @functools.lru_cache(maxsize=8192)
    def is_subtype(self, sub: str, super_: str) -> bool:
        sub_c = self.canonical_name(sub)
        super_c = self.canonical_name(super_)

        if super_c in ("any", "Any", "object", "DataObject", "*", "top") or sub_c in ("any", "Any") or sub_c == super_c:
            return True

        if sub_c not in self._parents:
            return False

        visited = set()
        queue = deque([sub_c])
        while queue:
            curr = queue.popleft()
            if curr == super_c:
                return True
            visited.add(curr)
            for parent in self._parents.get(curr, []):
                if parent not in visited:
                    queue.append(parent)

        return False


@dataclass(frozen=True, slots=True)
class AlgebraicSignature:
    type_name: str
    state: str = "any"
    qualifiers: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)

    def is_top(self) -> bool:
        return self.type_name.lower() in ("any", "*", "top", "anyobject", "object")

    def unifies_with(self, other: Any) -> bool:
        """
        Returns True if `self` (producer output) unifies with `other` (consumer input).
        """
        if hasattr(other, "signature") and isinstance(other.signature, AlgebraicSignature):
            other_sig = other.signature
        elif isinstance(other, AlgebraicSignature):
            other_sig = other
        else:
            return False

        # 1. State check: if both specify a concrete state, they must match
        if self.state != "any" and other_sig.state != "any":
            if self.state.lower() != other_sig.state.lower():
                return False

        # 2. Type satisfaction check
        if other_sig.is_top():
            return True  # Consumer accepts anything
        if self.is_top():
            return False # Producer is untyped, cannot guarantee concrete type

        registry = TypeRegistry.get_instance()
        if not registry.is_subtype(self.type_name, other_sig.type_name):
            return False

        if other_sig.qualifiers and not other_sig.qualifiers.issubset(self.qualifiers):
            return False

        return True


class PortSignature:
    __slots__ = ["name", "signature", "required", "default_value", "doc"]

    def __init__(
        self,
        name: str = "",
        signature: Union[AlgebraicSignature, str, Any] = "any",
        required: bool = True,
        default_value: Optional[str] = None,
        doc: str = "",
        **kwargs
    ):
        if isinstance(signature, str):
            t_name = name if (name and (name[0].isupper() or name in ("str", "int", "float", "bool", "dict", "list", "bytes", "DataFrame", "Mat", "ndarray", "Any", "AnyObject"))) else "str"
            self.name = name
            self.signature = AlgebraicSignature(type_name=t_name, state=signature)
        elif isinstance(signature, AlgebraicSignature):
            self.name = name
            self.signature = signature
        elif hasattr(signature, "signature") and isinstance(signature.signature, AlgebraicSignature):
            self.name = name or getattr(signature, "name", "")
            self.signature = signature.signature
        else:
            self.name = name
            self.signature = AlgebraicSignature("any", "any")

        self.required = required
        self.default_value = default_value
        self.doc = doc

    @property
    def type_name(self) -> str:
        return self.signature.type_name

    @property
    def state(self) -> str:
        return self.signature.state

    def is_top(self) -> bool:
        return self.signature.is_top()

    def unifies_with(self, other: Any) -> bool:
        if isinstance(other, PortSignature):
            return self.signature.unifies_with(other.signature)
        if isinstance(other, AlgebraicSignature):
            return self.signature.unifies_with(other)
        return False

    def __repr__(self) -> str:
        return f"PortSignature(name={self.name!r}, signature={self.signature!r}, required={self.required})"


class Cell(ABC):
    __slots__ = [
        "cell_id", "stage", "keywords", "cell_type",
        "inputs", "outputs", "domain_name", "node_type", "node_role",
        "dependencies", "code_template", "metadata_tags", "_db_path",
        "configuration_schema", "verified", "semantic_tags",
        "docstring", "enrichment_source", "enriched_at"
    ]

    def __init__(
        self,
        cell_id: str,
        stage: int = 2,
        keywords: Optional[Union[Set[str], List[str]]] = None,
        cell_type: str = "micro",
        inputs: Optional[Dict[str, Union[PortSignature, AlgebraicSignature, dict]]] = None,
        outputs: Optional[Dict[str, Union[PortSignature, AlgebraicSignature, dict]]] = None,
        domain_name: str = "",
        node_type: str = "function",
        node_role: str = "function",
        dependencies: Optional[List[str]] = None,
        code_template: str = "",
        metadata_tags: Optional[Dict[str, Any]] = None,
        db_path: str = "",
        configuration_schema: Optional[Dict[str, Any]] = None,
        verified: bool = False,
        semantic_tags: Optional[List[str]] = None,
        docstring: str = "",
        enrichment_source: Optional[str] = None,
        enriched_at: Optional[str] = None
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(k.lower() for k in keywords if len(str(k)) >= 3) if keywords else set()
        self.cell_type = cell_type
        self.domain_name = domain_name
        self.node_type = node_type
        self.node_role = str(node_role).lower() if node_role else "function"
        self.dependencies = dependencies or []
        self.code_template = code_template
        self.metadata_tags = metadata_tags or {}
        self._db_path = db_path
        self.configuration_schema = configuration_schema or {}
        self.verified = bool(verified)
        self.semantic_tags = list(semantic_tags) if semantic_tags else list(self.keywords)
        self.docstring = docstring or ""
        self.enrichment_source = enrichment_source
        self.enriched_at = enriched_at

        # Defensive normalization: accept PortSignature, AlgebraicSignature, or raw dicts
        self.inputs: Dict[str, PortSignature] = {}
        for k, v in (inputs or {}).items():
            if isinstance(v, PortSignature):
                self.inputs[k] = v
            elif isinstance(v, AlgebraicSignature):
                self.inputs[k] = PortSignature(name=k, signature=v)
            elif isinstance(v, dict):
                sig = AlgebraicSignature(
                    type_name=v.get("type_name", "any"),
                    state=v.get("state", "any"),
                    qualifiers=frozenset(tuple(q) for q in v.get("qualifiers", []))
                )
                self.inputs[k] = PortSignature(
                    name=k,
                    signature=sig,
                    required=v.get("required", True),
                    default_value=v.get("default_value"),
                    doc=v.get("doc", "")
                )
            else:
                self.inputs[k] = PortSignature(name=k, signature=AlgebraicSignature("any", "any"))

        self.outputs: Dict[str, PortSignature] = {}
        for k, v in (outputs or {}).items():
            if isinstance(v, PortSignature):
                self.outputs[k] = v
            elif isinstance(v, AlgebraicSignature):
                self.outputs[k] = PortSignature(name=k, signature=v)
            elif isinstance(v, dict):
                sig = AlgebraicSignature(
                    type_name=v.get("type_name", "any"),
                    state=v.get("state", "any"),
                    qualifiers=frozenset(tuple(q) for q in v.get("qualifiers", []))
                )
                self.outputs[k] = PortSignature(
                    name=k,
                    signature=sig,
                    required=v.get("required", True),
                    default_value=v.get("default_value"),
                    doc=v.get("doc", "")
                )
            else:
                self.outputs[k] = PortSignature(name=k, signature=AlgebraicSignature("any", "any"))

    @property
    def primary_input(self) -> PortSignature:
        """
        Identifies the main data port based on typestate and stage,
        independent of dictionary insertion order.
        """
        if not self.inputs:
            return PortSignature("input_data", AlgebraicSignature("any", "any"))

        # Stage 1 (Source): expects URI / path identifier
        if self.stage == 1:
            for p in self.inputs.values():
                if p.state in ("source_identifier", "filepath_read", "input_uri", "source_uri"):
                    return p
            return next(iter(self.inputs.values()))

        # Stage 2 & 3: prioritize container data types over parameter primitives
        data_types = {"DataFrame", "ndarray", "Mat", "Figure", "Graph", "Tensor", "Series", "Dataset"}
        for p in self.inputs.values():
            if p.type_name in data_types or p.state in ("raw", "cleaned", "sorted", "plotted", "scaled", "data_payload"):
                return p

        # Fallback to the first non-primitive port if available
        for p in self.inputs.values():
            if p.type_name not in ("str", "int", "float", "bool"):
                return p

        return next(iter(self.inputs.values()))

    def can_accept(self, sig: Union[AlgebraicSignature, PortSignature]) -> bool:
        """Returns True if this cell has any input port that unifies with `sig`."""
        target_sig = sig.signature if hasattr(sig, "signature") else sig
        for p in self.inputs.values():
            if target_sig.unifies_with(p.signature):
                return True
        return False

    @property
    def primary_output(self) -> PortSignature:
        if not self.outputs:
            return PortSignature("output_data", AlgebraicSignature("None", "any"))
        data_types = {"DataFrame", "ndarray", "Mat", "Figure", "Graph", "Tensor", "Series", "Dataset"}
        for p in self.outputs.values():
            if p.type_name in data_types or p.state in ("raw", "cleaned", "sorted", "plotted", "scaled", "data_payload", "filepath_written"):
                return p
        return next(iter(self.outputs.values()))

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.cell_id} "
            f"({self.domain_name}) {self.primary_input.signature} -> {self.primary_output.signature}>"
        )


class MicroCell(Cell):
    __slots__ = ()  # Inherit parent slots exactly; no extra __dict__

    def __init__(self, **kwargs):
        kwargs["cell_type"] = "micro"
        super().__init__(**kwargs)


class MacroCell(Cell):
    __slots__ = ("sub_cells", "algorithmic_steps")

    def __init__(
        self,
        sub_cells: Optional[List[str]] = None,
        algorithmic_steps: Optional[List[str]] = None,
        **kwargs
    ):
        kwargs["cell_type"] = "macro"
        super().__init__(**kwargs)
        self.sub_cells = sub_cells or []
        self.algorithmic_steps = algorithmic_steps or []


class LatticeOrchestrator:
    """
    Manages the live typed DAG using lazy on-demand bucket resolution.
    """
    def __init__(self, trees_directory="trees", active_domain="Python_Core"):
        self.trees_directory = trees_directory
        self.db_path = os.path.join(trees_directory, "lattice.db")
        self.active_domain = active_domain
        self.loaded_cells: Dict[str, Cell] = {}
        # Indexed bucket lookup tables for O(1) successor and predecessor retrieval
        self._cells_by_input: Dict[Tuple[str, str], List[Cell]] = {}
        self._cells_by_stage_input: Dict[Tuple[int, str, str], List[Cell]] = {}
        self._cells_by_output: Dict[Tuple[str, str], List[Cell]] = {}
        self._cells_by_keyword: Dict[str, List[Cell]] = {}
        self._lock = threading.RLock()

        self.load_from_database()
        self.build_topology()

    @property
    def cells(self) -> List[Cell]:
        return list(self.loaded_cells.values())

    def load_from_database(self, db_path: Optional[str] = None):
        with self._lock:
            if db_path is not None:
                self.db_path = db_path
            if not os.path.exists(self.db_path):
                logger.warning(f"[LATTICE] No SQLite DB found at {self.db_path}")
                return
    
            self.loaded_cells.clear()
    
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('nodes', 'cells')")
                tables = [row[0] for row in cursor.fetchall()]
    
                if "nodes" in tables:
                    cursor.execute("PRAGMA table_info(nodes)")
                    cols = [c[1] for c in cursor.fetchall()]
                    has_enrichment = "docstring" in cols

                    if has_enrichment:
                        cursor.execute("""
                            SELECT cell_id, domain_name, node_type, node_role, stage,
                                   keywords, input_type, input_state, output_type, output_state,
                                   code, dependencies, configuration_schema, verified,
                                   docstring, enrichment_source, enriched_at
                            FROM nodes
                        """)
                    else:
                        cursor.execute("""
                            SELECT cell_id, domain_name, node_type, node_role, stage,
                                   keywords, input_type, input_state, output_type, output_state,
                                   code, dependencies, configuration_schema, verified
                            FROM nodes
                        """)
                    rows = cursor.fetchall()
                    for row in rows:
                        if has_enrichment:
                            (cell_id, domain_name, node_type, node_role, stage,
                             keywords_json, in_type, in_state, out_type, out_state,
                             code, deps_json, config_json, verified,
                             doc_str, enrich_src, enrich_at) = row
                        else:
                            (cell_id, domain_name, node_type, node_role, stage,
                             keywords_json, in_type, in_state, out_type, out_state,
                             code, deps_json, config_json, verified) = row
                            doc_str, enrich_src, enrich_at = "", None, None
    
                        try:
                            keywords = set(json.loads(keywords_json)) if keywords_json else set()
                        except Exception:
                            keywords = set()
    
                        try:
                            dependencies = json.loads(deps_json) if deps_json else []
                        except Exception:
                            dependencies = []
    
                        try:
                            configuration_schema = json.loads(config_json) if config_json else {}
                        except Exception:
                            configuration_schema = {}
    
                        in_sig = AlgebraicSignature(type_name=in_type or "any", state=in_state or "any")
                        out_sig = AlgebraicSignature(type_name=out_type or "None", state=out_state or "any")
    
                        inputs: Dict[str, PortSignature] = {}
                        outputs: Dict[str, PortSignature] = {}
                        if isinstance(configuration_schema, dict) and configuration_schema:
                            raw_in = configuration_schema.get("inputs", configuration_schema)
                            if isinstance(raw_in, dict):
                                for p_name, p_val in raw_in.items():
                                    if p_name in ("inputs", "outputs"):
                                        continue
                                    if isinstance(p_val, dict):
                                        p_type = p_val.get("type_name", p_val.get("type", in_type or "any"))
                                        p_state = p_val.get("state", in_state or "any")
                                        inputs[p_name] = PortSignature(
                                            name=p_name,
                                            signature=AlgebraicSignature(type_name=str(p_type), state=str(p_state)),
                                            required=p_val.get("required", True),
                                            default_value=p_val.get("default_value", p_val.get("default")),
                                            doc=p_val.get("doc", p_val.get("param_doc", ""))
                                        )
                                    elif isinstance(p_val, str):
                                        inputs[p_name] = PortSignature(
                                            name=p_name,
                                            signature=AlgebraicSignature(type_name=p_val, state="any")
                                        )
                            raw_out = configuration_schema.get("outputs", {})
                            if isinstance(raw_out, dict):
                                for p_name, p_val in raw_out.items():
                                    if isinstance(p_val, dict):
                                        p_type = p_val.get("type_name", out_type or "any")
                                        p_state = p_val.get("state", out_state or "any")
                                        outputs[p_name] = PortSignature(
                                            name=p_name,
                                            signature=AlgebraicSignature(type_name=str(p_type), state=str(p_state))
                                        )
                                    elif isinstance(p_val, str):
                                        outputs[p_name] = PortSignature(
                                            name=p_name,
                                            signature=AlgebraicSignature(type_name=p_val, state="any")
                                        )
                        elif isinstance(configuration_schema, list) and configuration_schema:
                            for p_val in configuration_schema:
                                if isinstance(p_val, dict) and "name" in p_val:
                                    p_name = p_val["name"]
                                    p_type = p_val.get("type_name", p_val.get("type", in_type or "any"))
                                    p_state = p_val.get("state", in_state or "any")
                                    inputs[p_name] = PortSignature(
                                        name=p_name,
                                        signature=AlgebraicSignature(type_name=str(p_type), state=str(p_state)),
                                        required=p_val.get("required", True),
                                        default_value=p_val.get("default_value", p_val.get("default")),
                                        doc=p_val.get("doc", p_val.get("param_doc", ""))
                                    )
    
                        if not inputs:
                            inputs = {"input_data": PortSignature(name="input_data", signature=in_sig)}
                        if not outputs:
                            outputs = {"output_data": PortSignature(name="output_data", signature=out_sig)}
    
                        is_macro = node_type == "macro" or str(node_role).lower() == "macro"
                        cell_cls = MacroCell if is_macro else MicroCell
    
                        cell = cell_cls(
                            cell_id=cell_id,
                            stage=stage or 1,
                            keywords=keywords,
                            inputs=inputs,
                            outputs=outputs,
                            domain_name=domain_name or self.active_domain,
                            node_type="macro" if is_macro else (node_type or "function"),
                            node_role=str(node_role).lower() if node_role else "function",
                            dependencies=dependencies,
                            code_template=code or "",
                            db_path=self.db_path,
                            configuration_schema=configuration_schema,
                            verified=bool(verified),
                            docstring=doc_str or "",
                            enrichment_source=enrich_src,
                            enriched_at=enrich_at
                        )
                        self.loaded_cells[cell.cell_id] = cell
    
                elif "cells" in tables:
                    cursor.execute("""
                        SELECT cell_id, stage, input_type, input_state, output_type, output_state,
                               code_template, configuration_schema, dependencies
                        FROM cells
                    """)
                    rows = cursor.fetchall()
                    for row in rows:
                        (cell_id, stage, in_type, in_state, out_type, out_state,
                         code, config_json, deps_json) = row
    
                        try:
                            dependencies = json.loads(deps_json) if deps_json else []
                        except Exception:
                            dependencies = []
    
                        try:
                            configuration_schema = json.loads(config_json) if config_json else {}
                        except Exception:
                            configuration_schema = {}
    
                        in_sig = AlgebraicSignature(type_name=in_type or "any", state=in_state or "any")
                        out_sig = AlgebraicSignature(type_name=out_type or "None", state=out_state or "any")
    
                        inputs: Dict[str, PortSignature] = {}
                        outputs: Dict[str, PortSignature] = {}
    
                        if isinstance(configuration_schema, dict) and ("inputs" in configuration_schema or "outputs" in configuration_schema):
                            raw_in = configuration_schema.get("inputs", {})
                            for p_name, p_val in raw_in.items():
                                p_type = p_val.get("type_name", in_type or "any")
                                p_state = p_val.get("state", in_state or "any")
                                inputs[p_name] = PortSignature(
                                    name=p_name,
                                    signature=AlgebraicSignature(type_name=str(p_type), state=str(p_state)),
                                    required=p_val.get("required", True),
                                    default_value=p_val.get("default_value", p_val.get("default"))
                                )
                            raw_out = configuration_schema.get("outputs", {})
                            for p_name, p_val in raw_out.items():
                                p_type = p_val.get("type_name", out_type or "any")
                                p_state = p_val.get("state", out_state or "any")
                                outputs[p_name] = PortSignature(
                                    name=p_name,
                                    signature=AlgebraicSignature(type_name=str(p_type), state=str(p_state))
                                )
    
                        if not inputs:
                            inputs = {"input_data": PortSignature(name="input_data", signature=in_sig)}
                        if not outputs:
                            outputs = {"output_data": PortSignature(name="output_data", signature=out_sig)}
    
                        keywords = set(re.findall(r'[a-zA-Z0-9]+', cell_id.lower())) - {"cell", "default", "broken", "distractor"}
                        for p in inputs.values():
                            if p.state != "any":
                                keywords.update(re.findall(r'[a-zA-Z0-9]+', p.state.lower()))
                            if p.name:
                                keywords.update(re.findall(r'[a-zA-Z0-9]+', p.name.lower()))
                        for p in outputs.values():
                            if p.state != "any":
                                keywords.update(re.findall(r'[a-zA-Z0-9]+', p.state.lower()))
    
                        cell = MicroCell(
                            cell_id=cell_id,
                            stage=stage or 1,
                            keywords=keywords,
                            inputs=inputs,
                            outputs=outputs,
                            domain_name="pandas",
                            node_type="function",
                            node_role="function",
                            dependencies=dependencies,
                            code_template=code or "",
                            db_path=self.db_path,
                            configuration_schema=configuration_schema
                        )
                        self.loaded_cells[cell.cell_id] = cell
    
                conn.close()
                logger.info(f"[LATTICE] Loaded {len(self.loaded_cells)} cells from database.")
            except Exception as e:
                logger.error(f"[LATTICE] Database load error: {e}")

    def build_reachability_matrix(self) -> None:
        """
        Precomputes transitive reachability between typestates as bitsets
        for instant O(1) dead-end pruning during A* expansion.
        """
        with self._lock:
            reg = TypeRegistry.get_instance()
            all_types = list(set(list(reg.poset.keys()) + [
                "DataFrame", "ndarray", "Mat", "Figure", "Graph", "Tensor", "Series", "Dataset",
                "str", "int", "float", "bool", "dict", "list", "None", "any", "Image", "AudioData"
            ]))
            self.type_to_idx = {t: i for i, t in enumerate(all_types)}
            self.reachability_bitsets: Dict[str, int] = {}

            # Build direct transition adjacency
            direct_adj: Dict[str, Set[str]] = {t: set() for t in all_types}
            for cell in self.loaded_cells.values():
                in_sig = cell.primary_input
                out_sig = cell.primary_output
                if in_sig and out_sig:
                    in_t = in_sig.type_name
                    out_t = out_sig.type_name
                    direct_adj.setdefault(in_t, set()).add(out_t)
                    if in_t == "Mat":
                        direct_adj.setdefault("ndarray", set()).add(out_t)
                    if in_t == "Series":
                        direct_adj.setdefault("DataFrame", set()).add(out_t)

            # Transitive closure (BFS reachability)
            for t in all_types:
                reachable = set([t])
                queue = deque([t])
                while queue:
                    curr = queue.popleft()
                    for nxt in direct_adj.get(curr, []):
                        if nxt not in reachable:
                            reachable.add(nxt)
                            queue.append(nxt)

                bitset = 0
                for r in reachable:
                    if r in self.type_to_idx:
                        bitset |= (1 << self.type_to_idx[r])
                self.reachability_bitsets[t] = bitset
            logger.info(f"[LATTICE] Reachability bitset matrix precomputed for {len(all_types)} types.")

    def can_reach_type(self, source_type: str, target_type: str) -> bool:
        """O(1) Bitwise Reachability Check."""
        if source_type == target_type or target_type in ("any", "None", "str", "object"):
            return True
        if not hasattr(self, "reachability_bitsets") or not self.reachability_bitsets:
            return True
        if source_type not in self.reachability_bitsets or target_type not in self.type_to_idx:
            return True  # Permissive fallback for dynamic types
        target_bit = 1 << self.type_to_idx[target_type]
        return bool(self.reachability_bitsets[source_type] & target_bit)

    def build_topology(self):
        """Builds O(1) indexed lookup tables for both inputs, stages, and outputs."""
        with self._lock:
            self._cells_by_input.clear()
            self._cells_by_stage_input.clear()
            self._cells_by_output.clear()
            self._cells_by_keyword.clear()
            for cell in self.loaded_cells.values():
                in_sig = cell.primary_input
                self._cells_by_input.setdefault((in_sig.type_name, in_sig.state), []).append(cell)
                self._cells_by_stage_input.setdefault((cell.stage, in_sig.type_name, in_sig.state), []).append(cell)
                out_sig = cell.primary_output
                self._cells_by_output.setdefault((out_sig.type_name, out_sig.state), []).append(cell)
                for kw in cell.keywords:
                    self._cells_by_keyword.setdefault(kw.lower(), []).append(cell)
            self.build_reachability_matrix()
            logger.info(
                f"[LATTICE] Indexed {len(self.loaded_cells)} cells into "
                f"{len(self._cells_by_input)} input buckets and {len(self._cells_by_output)} output buckets."
            )

    def inject_cell(self, cell: Cell):
        """Thread-safe incremental addition of a synthesized cell into the orchestrator."""
        with self._lock:
            self.loaded_cells[cell.cell_id] = cell
            in_sig = cell.primary_input
            self._cells_by_input.setdefault((in_sig.type_name, in_sig.state), []).append(cell)
            self._cells_by_stage_input.setdefault((cell.stage, in_sig.type_name, in_sig.state), []).append(cell)
            out_sig = cell.primary_output
            self._cells_by_output.setdefault((out_sig.type_name, out_sig.state), []).append(cell)
            for kw in cell.keywords:
                self._cells_by_keyword.setdefault(kw.lower(), []).append(cell)

    def register_cell(self, cell: Cell):
        """Alias for inject_cell."""
        self.inject_cell(cell)

    def get_successors_for_sig(self, sig: Union[AlgebraicSignature, PortSignature], stage: Optional[int] = None) -> List[Cell]:
        """Returns type-compatible downstream successor cells in O(1) dictionary lookups."""
        sig_val = sig.signature if hasattr(sig, "signature") else sig
        registry = TypeRegistry.get_instance()
        applicable_types = {sig_val.type_name, "any", "AnyObject", "object", "*"}
        curr_type = sig_val.type_name
        with registry._lock:
            ancestors = set()
            queue = deque(registry._parents.get(curr_type, []))
            while queue:
                p = queue.popleft()
                if p not in ancestors:
                    ancestors.add(p)
                    queue.extend(registry._parents.get(p, []))
            applicable_types.update(ancestors)

        applicable_states = {sig_val.state, "any"}
        candidates: List[Cell] = []
        seen_ids: Set[str] = set()

        with self._lock:
            for t in applicable_types:
                for s in applicable_states:
                    if stage is not None:
                        bucket = self._cells_by_stage_input.get((stage, t, s), [])
                    else:
                        bucket = self._cells_by_input.get((t, s), [])
                    for cell in bucket:
                        if cell.cell_id not in seen_ids:
                            seen_ids.add(cell.cell_id)
                            candidates.append(cell)

        return candidates

    def get_predecessors_for_sig(self, sig: AlgebraicSignature) -> List[Cell]:
        """Returns type-compatible upstream predecessor cells in O(1) dictionary lookups."""
        applicable_types = {sig.type_name, "any", "AnyObject", "object", "*"}
        applicable_states = {sig.state, "any"}
        candidates: List[Cell] = []
        seen_ids: Set[str] = set()

        with self._lock:
            for t in applicable_types:
                for s in applicable_states:
                    for cell in self._cells_by_output.get((t, s), []):
                        if cell.cell_id not in seen_ids:
                            if cell.primary_output.unifies_with(sig):
                                seen_ids.add(cell.cell_id)
                                candidates.append(cell)

        return candidates

    def get_neighbors(self, cell_id: str) -> List[Cell]:
        """Returns type-compatible downstream successor cells on-demand in O(1)."""
        cell = self.loaded_cells.get(cell_id)
        if not cell:
            return []
        return [c for c in self.get_successors_for_sig(cell.primary_output) if c.cell_id != cell_id]

    def get_all_available_cells(self) -> List[Cell]:
        """Returns all loaded cells in the active lattice."""
        return list(self.loaded_cells.values())

    def get_cell(self, cell_id: str) -> Optional[Cell]:
        """Safe accessor for a single cell by ID."""
        return self.loaded_cells.get(cell_id)
