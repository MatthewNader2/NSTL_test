"""
src/lattice.py - Neuro-Symbolic Topological Lattice (NSTL)
Domain-Agnostic Typed Lattice, Algebraic Signatures, and Modular Graph Topology.

Conforms strictly to Sections 3.1-3.2 of the NSTL paper:
  Every primitive is represented as a node v = (f, tau_in, tau_out).
  Nodes are organized into a lattice (directed graph) where an edge from
  node u to node v exists iff unify(tau_out(u), tau_in(v)) != bottom.
"""

from __future__ import annotations
import functools
import json
import os
import re
import sqlite3
import threading
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, FrozenSet, Union

from log_config import get_logger

logger = get_logger('lattice')


class TypeRegistry:
    """
    Dynamic Poset Type Hierarchy (T, <=).
    Maintains a directed acyclic graph of subtype relationships.
    Types are registered dynamically from trees without hardcoded domain dependencies.
    """
    _instance: Optional[TypeRegistry] = None
    _lock = threading.RLock()

    def __init__(self):
        self._parents: Dict[str, Set[str]] = {}
        self._aliases: Dict[str, str] = {}
        self._register_primitive_types()

    @classmethod
    def get_instance(cls) -> TypeRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    @property
    def poset(self) -> Dict[str, Set[str]]:
        return self._parents

    def _register_primitive_types(self):
        """Universal foundational types common across computing systems."""
        self.register_type("any")
        self.register_type("object", super_type="any")
        self.register_type("numeric", super_type="object")
        self.register_type("None", super_type="any")
        self.register_type("str", super_type="object")
        self.register_type("int", super_type="numeric")
        self.register_type("float", super_type="numeric")
        self.register_type("bool", super_type="int")
        self.register_type("list", super_type="object")
        self.register_type("dict", super_type="object")
        self.register_type("tuple", super_type="object")

    def register_type(self, type_name: str, super_type: Optional[str] = None):
        """Registers a type and optionally declares its supertype in the poset."""
        name = str(type_name).strip()
        if not name:
            return
        if name not in self._parents:
            self._parents[name] = set()
        if super_type:
            super_name = str(super_type).strip()
            if super_name and super_name != name:
                if super_name not in self._parents:
                    self._parents[super_name] = set()
                self._parents[name].add(super_name)
        try:
            self.is_subtype.cache_clear()
        except AttributeError:
            pass

    def register_alias(self, alias: str, canonical: str):
        self._aliases[alias.strip().lower()] = canonical.strip()

    def canonical_name(self, type_name: str) -> str:
        if not type_name:
            return "any"
        clean = str(type_name).strip()
        return self._aliases.get(clean.lower(), clean)

    @functools.lru_cache(maxsize=16384)
    def is_subtype(self, sub: str, super_: str) -> bool:
        """
        Computes poset partial order: returns True iff sub <= super_.
        Wildcards ('any', 'object', '*', 'top') are Top types that subsume all types.
        """
        sub_c = self.canonical_name(sub)
        super_c = self.canonical_name(super_)

        if super_c.lower() in ("any", "object", "*", "top", "unknown"):
            return True
        if sub_c.lower() in ("any", "top", "*"):
            return False
        if sub_c.lower() == super_c.lower():
            return True

        if sub_c not in self._parents:
            return False

        visited = set()
        queue = deque([sub_c])
        while queue:
            curr = queue.popleft()
            if curr.lower() == super_c.lower():
                return True
            visited.add(curr)
            for parent in self._parents.get(curr, []):
                if parent not in visited:
                    queue.append(parent)

        return False

    def is_container_type(self, type_name: str) -> bool:
        """
        True if type is non-primitive (i.e. not a basic scalar int/float/bool/str/None).
        """
        canonical = self.canonical_name(type_name).lower()
        primitive_types = {"int", "float", "bool", "str", "none"}
        return canonical not in primitive_types


def is_subtype(sub: str, parent: str) -> bool:
    return TypeRegistry.get_instance().is_subtype(sub, parent)


def canonical_type_name(type_name: str) -> str:
    return TypeRegistry.get_instance().canonical_name(type_name)


@dataclass(frozen=True, slots=True)
class AlgebraicSignature:
    """
    Formal typestate signature: tau = (type_name, state, qualifiers).
    Conforms to Section 3.1 of the NSTL paper.
    """
    type_name: str = "any"
    state: str = "any"
    qualifiers: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)

    @classmethod
    def from_string(cls, type_name: str, state: str = "any") -> "AlgebraicSignature":
        return cls(type_name=type_name, state=state)

    def is_top(self) -> bool:
        return self.type_name.lower() in ("any", "*", "top", "object", "unknown")

    def unifies_with(self, other: Any) -> bool:
        """
        Evaluates whether producer output `self` can satisfy consumer input `other`.
        Rules:
          1. Consumer Top accepts any producer type.
          2. Non-top consumer rejects top producer.
          3. If both declare non-'any' state, states must match (case-insensitive).
          4. Producer type must be a subtype of consumer type in the Type Poset.
          5. Consumer qualifiers must be a subset of producer qualifiers.
        """
        if hasattr(other, "signature") and isinstance(other.signature, AlgebraicSignature):
            other_sig = other.signature
        elif isinstance(other, AlgebraicSignature):
            other_sig = other
        else:
            return False

        # State compatibility check
        if self.state != "any" and other_sig.state != "any":
            if self.state.lower() != other_sig.state.lower():
                return False

        # Consumer accepts anything
        if other_sig.is_top():
            return True
        # Producer is untyped wildcard, cannot guarantee concrete type requirement
        if self.is_top():
            return False

        # Poset subtyping
        registry = TypeRegistry.get_instance()
        if not registry.is_subtype(self.type_name, other_sig.type_name):
            return False

        # Qualifier satisfaction
        if other_sig.qualifiers and not other_sig.qualifiers.issubset(self.qualifiers):
            return False

        return True

    def matches(self, other: Any) -> bool:
        return self.unifies_with(other)


class PortSignature:
    """Named port carrying an AlgebraicSignature typestate."""
    __slots__ = ["name", "signature", "required", "default_value", "doc", "domain"]

    def __init__(
        self,
        name: str = "",
        signature: Union[AlgebraicSignature, str, Any] = "any",
        required: bool = True,
        default_value: Optional[Any] = None,
        doc: str = "",
        domain: str = "",
        **kwargs
    ):
        self.name = str(name)
        self.domain = str(domain or kwargs.get("domain", ""))
        if isinstance(signature, AlgebraicSignature):
            self.signature = signature
        elif hasattr(signature, "signature") and isinstance(signature.signature, AlgebraicSignature):
            self.signature = signature.signature
        elif isinstance(signature, str):
            if "name" in kwargs:
                self.name = kwargs["name"]
                self.signature = AlgebraicSignature(type_name=signature, state=kwargs.get("state", "any"))
            elif signature != "any":
                self.name = kwargs.get("name", f"port_{name}")
                self.signature = AlgebraicSignature(type_name=name, state=signature)
            else:
                self.signature = AlgebraicSignature(type_name=name if name else "any", state="any")
        else:
            self.signature = AlgebraicSignature("any", "any")

        self.required = bool(required)
        self.default_value = default_value
        self.doc = str(doc or "")

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
        return f"Port({self.name}: {self.signature.type_name}[{self.signature.state}], req={self.required})"


class Cell(ABC):
    """
    Mathematical Lattice Node: v = (f, tau_in, tau_out).
    Independent of domain, language, or execution runtime.
    """
    __slots__ = [
        "cell_id", "stage", "keywords", "cell_type",
        "inputs", "outputs", "slots", "domain_name", "node_type", "node_role",
        "dependencies", "code_template", "metadata_tags",
        "configuration_schema", "verified", "semantic_tags",
        "docstring", "enrichment_source", "enriched_at",
        "source_priority", "source_provenance",
        "_primary_input", "_primary_output", "_token_set"
    ]

    def __init__(
        self,
        cell_id: str,
        stage: int = 2,
        keywords: Optional[Union[Set[str], List[str]]] = None,
        cell_type: str = "micro",
        inputs: Optional[Dict[str, Union[PortSignature, AlgebraicSignature, dict]]] = None,
        outputs: Optional[Dict[str, Union[PortSignature, AlgebraicSignature, dict]]] = None,
        slots: Optional[Dict[str, Any]] = None,
        domain_name: str = "generic",
        node_type: str = "function",
        node_role: str = "function",
        dependencies: Optional[List[str]] = None,
        code_template: str = "",
        metadata_tags: Optional[Dict[str, Any]] = None,
        configuration_schema: Optional[Dict[str, Any]] = None,
        verified: bool = False,
        semantic_tags: Optional[List[str]] = None,
        docstring: str = "",
        enrichment_source: Optional[str] = None,
        enriched_at: Optional[str] = None,
        source_priority: int = 100,
        source_provenance: Optional[str] = "unknown"
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(str(k).lower() for k in keywords if len(str(k)) >= 3) if keywords else set()
        self.cell_type = cell_type
        self.domain_name = domain_name
        self.node_type = node_type
        self.node_role = str(node_role).lower() if node_role else "function"
        self.slots = slots or {}
        self.dependencies = dependencies or []
        self.code_template = code_template
        self.metadata_tags = metadata_tags or {}
        self.configuration_schema = configuration_schema or {}
        self.source_priority = int(source_priority) if source_priority is not None else 100
        self.source_provenance = str(source_provenance) if source_provenance else "unknown"
        self.verified = bool(verified)
        self.semantic_tags = list(semantic_tags) if semantic_tags else list(self.keywords)
        self.docstring = docstring or ""
        self.enrichment_source = enrichment_source
        self.enriched_at = enriched_at

        self._primary_input = None
        self._primary_output = None

        # Normalize inputs into Dict[str, PortSignature]
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
                    doc=v.get("doc", ""),
                    domain=v.get("domain", "")
                )
            else:
                self.inputs[k] = PortSignature(name=k, signature=AlgebraicSignature("any", "any"))

        # Normalize outputs into Dict[str, PortSignature]
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
                    doc=v.get("doc", ""),
                    domain=v.get("domain", "")
                )
            else:
                self.outputs[k] = PortSignature(name=k, signature=AlgebraicSignature("any", "any"))

        self._token_set: Optional[Set[str]] = None

        # Register types automatically in TypeRegistry
        registry = TypeRegistry.get_instance()
        for p in self.inputs.values():
            registry.register_type(p.type_name)
        for p in self.outputs.values():
            registry.register_type(p.type_name)

    @property
    def token_set(self) -> Set[str]:
        if self._token_set is None:
            try:
                from .tokenizer import CellTokenizer
            except (ImportError, ValueError):
                from tokenizer import CellTokenizer
            toks = CellTokenizer.tokenize_cell(self.cell_id, self.keywords)
            for p in self.inputs:
                toks.update(CellTokenizer.tokenize_identifier(p))
            for p in self.outputs:
                toks.update(CellTokenizer.tokenize_identifier(p))
            self._token_set = toks
        return self._token_set

    @property
    def primary_input(self) -> PortSignature:
        """Identifies the primary data-bearing input port."""
        if self._primary_input is not None:
            return self._primary_input

        if not self.inputs:
            res = PortSignature("input_data", AlgebraicSignature("any", "any"))
            self._primary_input = res
            return res

        # Prefer non-primitive / container data ports over auxiliary scalar parameters
        registry = TypeRegistry.get_instance()
        for p in self.inputs.values():
            if registry.is_container_type(p.type_name):
                self._primary_input = p
                return p

        # Fallback to the first declared input port
        res = next(iter(self.inputs.values()))
        self._primary_input = res
        return res

    @property
    def primary_output(self) -> PortSignature:
        """Identifies the primary data-bearing output port."""
        if self._primary_output is not None:
            return self._primary_output

        if not self.outputs:
            res = PortSignature("output_data", AlgebraicSignature("None", "any"))
            self._primary_output = res
            return res

        # Prioritize container / non-primitive data types
        registry = TypeRegistry.get_instance()
        for p in self.outputs.values():
            if registry.is_container_type(p.type_name):
                self._primary_output = p
                return p

        res = next(iter(self.outputs.values()))
        self._primary_output = res
        return res

    @property
    def is_public_morphism(self) -> bool:
        """
        Evaluates whether the cell represents a public morphism in domain category C.
        Excludes private/internal implementation artifacts (leading underscores, dunder methods).
        """
        if "__" in self.cell_id:
            return False
        for dep in self.dependencies:
            if "._" in dep or "import _" in dep:
                return False
        return True

    def can_accept(self, sig: Union[AlgebraicSignature, PortSignature]) -> bool:
        """True iff this cell has an input port that unifies with `sig`."""
        target_sig = sig.signature if hasattr(sig, "signature") else sig
        for p in self.inputs.values():
            if target_sig.unifies_with(p.signature):
                return True
        return False

    def __repr__(self) -> str:
        in_str = f"{self.primary_input.type_name}[{self.primary_input.state}]"
        out_str = f"{self.primary_output.type_name}[{self.primary_output.state}]"
        return f"<{self.__class__.__name__} {self.cell_id} ({self.domain_name}) {in_str} -> {out_str}>"


class MicroCell(Cell):
    """Primitive node in the lattice."""
    __slots__ = ()

    def __init__(self, **kwargs):
        kwargs["cell_type"] = "micro"
        super().__init__(**kwargs)


class MacroCell(Cell):
    """Higher-level hierarchical composite node in the lattice (Section 3.1)."""
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
    Mathematical Lattice Topology G = (V, E).
    Maintains nodes V and allows loading/unloading modular knowledge trees.
    An edge (u, v) exists iff u.primary_output unifies with v's accepting input port.
    """
    def __init__(self, trees_directory: str = "trees", active_domain: str = "all"):
        self.trees_directory = trees_directory
        self.db_path = os.path.join(trees_directory, "lattice.db")
        self.active_domain = active_domain
        self.loaded_cells: Dict[str, Cell] = {}
        self._adjacency: Dict[str, List[str]] = {}
        self._reverse_adjacency: Dict[str, List[str]] = {}
        self._cells_by_input: Dict[Tuple[str, str], List[Cell]] = {}
        self._cells_by_output: Dict[Tuple[str, str], List[Cell]] = {}
        self._lock = threading.RLock()

        if os.path.exists(self.db_path):
            self.load_from_database(self.db_path)
        else:
            self.load_all_json_trees()
        self.build_topology()

    @property
    def cells(self) -> List[Cell]:
        return list(self.loaded_cells.values())

    def load_tree_file(self, json_path: str):
        """Loads an arbitrary domain tree from JSON without engine hardcodes."""
        with self._lock:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                raw_cells = data
                domain = Path(json_path).stem
            else:
                domain = data.get("domain", "generic")
                raw_cells = data.get("cells", [])

            for c_dict in raw_cells:
                cell = MicroCell(
                    cell_id=c_dict.get("cell_id"),
                    stage=c_dict.get("stage", 2),
                    keywords=c_dict.get("keywords", []),
                    inputs=c_dict.get("inputs", {}),
                    outputs=c_dict.get("outputs", {}),
                    domain_name=c_dict.get("domain_name") or domain,
                    node_type=c_dict.get("node_type", "function"),
                    node_role=c_dict.get("node_role", "function"),
                    slots=c_dict.get("slots", {}),
                    dependencies=c_dict.get("dependencies", []),
                    code_template=c_dict.get("code_template", ""),
                    verified=c_dict.get("verified", True),
                    semantic_tags=c_dict.get("semantic_tags", []),
                    docstring=c_dict.get("docstring", ""),
                    source_priority=c_dict.get("source_priority", 100),
                )
                if cell.is_public_morphism:
                    self.loaded_cells[cell.cell_id] = cell
            logger.info(f"[LATTICE] Loaded {len(raw_cells)} nodes from tree: {json_path} (domain: {domain})")

    def unload_tree(self, domain: str):
        """Removes all nodes belonging to a tree domain."""
        with self._lock:
            to_remove = [cid for cid, c in self.loaded_cells.items() if c.domain_name == domain]
            for cid in to_remove:
                del self.loaded_cells[cid]
            logger.info(f"[LATTICE] Unloaded {len(to_remove)} nodes for domain: {domain}")

    def load_all_json_trees(self):
        """Loads all JSON trees located in trees_directory."""
        if not os.path.exists(self.trees_directory):
            return
        for fname in os.listdir(self.trees_directory):
            if fname.endswith(".json"):
                self.load_tree_file(os.path.join(self.trees_directory, fname))

    def load_from_database(self, db_path: Optional[str] = None):
        """Loads nodes from the compiled SQLite database."""
        with self._lock:
            if db_path is not None:
                self.db_path = db_path
            if not os.path.exists(self.db_path):
                logger.warning(f"[LATTICE] Database not found at {self.db_path}")
                return

            self.loaded_cells.clear()
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('nodes', 'cells')")
                tables = [row[0] for row in cursor.fetchall()]

                if "nodes" in tables:
                    cursor.execute("PRAGMA table_info(nodes)")
                    col_names = {row[1] for row in cursor.fetchall()}

                    doc_sel = "docstring" if "docstring" in col_names else "'' AS docstring"
                    prio_sel = "source_priority" if "source_priority" in col_names else "100 AS source_priority"
                    role_sel = "node_role" if "node_role" in col_names else "'function' AS node_role"
                    type_sel = "node_type" if "node_type" in col_names else "'function' AS node_type"
                    ver_sel = "verified" if "verified" in col_names else "1 AS verified"
                    dom_sel = "domain_name" if "domain_name" in col_names else "'' AS domain_name"
                    deps_sel = "dependencies" if "dependencies" in col_names else "'' AS dependencies"
                    cfg_sel = "configuration_schema" if "configuration_schema" in col_names else "'' AS configuration_schema"

                    cursor.execute(f"""
                        SELECT cell_id, {dom_sel}, {type_sel}, {role_sel}, stage,
                               keywords, input_type, input_state, output_type, output_state,
                               code, {deps_sel}, {cfg_sel}, {ver_sel}, {doc_sel}, {prio_sel}
                        FROM nodes
                    """)
                    for row in cursor.fetchall():
                        (cell_id, domain_name, node_type, node_role, stage,
                         keywords_json, in_type, in_state, out_type, out_state,
                         code, deps_json, config_json, verified, doc_str, source_priority) = row

                        try:
                            keywords = set(json.loads(keywords_json)) if keywords_json else set()
                        except Exception:
                            keywords = set()
                        try:
                            deps = json.loads(deps_json) if deps_json else []
                        except Exception:
                            deps = []
                        try:
                            cfg = json.loads(config_json) if config_json else {}
                        except Exception:
                            cfg = {}

                        in_sig = AlgebraicSignature(type_name=in_type or "any", state=in_state or "any")
                        out_sig = AlgebraicSignature(type_name=out_type or "None", state=out_state or "any")

                        inputs: Dict[str, PortSignature] = {}
                        outputs: Dict[str, PortSignature] = {}

                        if isinstance(cfg, dict) and ("inputs" in cfg or "outputs" in cfg):
                            for p_name, p_val in cfg.get("inputs", {}).items():
                                if isinstance(p_val, dict):
                                    inputs[p_name] = PortSignature(
                                        name=p_name,
                                        signature=AlgebraicSignature(
                                            type_name=str(p_val.get("type_name", in_type or "any")),
                                            state=str(p_val.get("state", in_state or "any")),
                                            qualifiers=frozenset(tuple(q) for q in p_val.get("qualifiers", []))
                                        ),
                                        required=p_val.get("required", True),
                                        default_value=p_val.get("default_value"),
                                        domain=p_val.get("domain", "")
                                    )
                            for p_name, p_val in cfg.get("outputs", {}).items():
                                if isinstance(p_val, dict):
                                    outputs[p_name] = PortSignature(
                                        name=p_name,
                                        signature=AlgebraicSignature(
                                            type_name=str(p_val.get("type_name", out_type or "any")),
                                            state=str(p_val.get("state", out_state or "any")),
                                            qualifiers=frozenset(tuple(q) for q in p_val.get("qualifiers", []))
                                        ),
                                        required=p_val.get("required", True),
                                        default_value=p_val.get("default_value"),
                                        domain=p_val.get("domain", "")
                                    )

                        if not inputs:
                            inputs = {"input_data": PortSignature("input_data", in_sig)}
                        if not outputs:
                            outputs = {"output_data": PortSignature("output_data", out_sig)}

                        is_macro = str(node_type).lower() in ("macro", "higher_order") or str(node_role).lower() in ("macro", "higher_order")
                        cls = MacroCell if is_macro else MicroCell

                        cell = cls(
                            cell_id=cell_id,
                            stage=stage or 2,
                            keywords=keywords,
                            inputs=inputs,
                            outputs=outputs,
                            slots=cfg.get("slots", {}),
                            domain_name=domain_name or "generic",
                            node_type="macro" if is_macro else (node_type or "function"),
                            node_role=str(node_role).lower() if node_role else "function",
                            dependencies=deps,
                            code_template=code or "",
                            verified=bool(verified),
                            docstring=doc_str or "",
                            source_priority=int(source_priority) if source_priority is not None else 100
                        )
                        if cell.is_public_morphism:
                            self.loaded_cells[cell.cell_id] = cell

                elif "cells" in tables:
                    cursor.execute("""
                        SELECT cell_id, stage, input_type, input_state, output_type, output_state,
                               code_template, configuration_schema, dependencies
                        FROM cells
                    """)
                    for row in cursor.fetchall():
                        (cell_id, stage, in_type, in_state, out_type, out_state,
                         code, config_json, deps_json) = row

                        try:
                            deps = json.loads(deps_json) if deps_json else []
                        except Exception:
                            deps = []
                        try:
                            cfg = json.loads(config_json) if config_json else {}
                        except Exception:
                            cfg = {}

                        in_sig = AlgebraicSignature(type_name=in_type or "any", state=in_state or "any")
                        out_sig = AlgebraicSignature(type_name=out_type or "None", state=out_state or "any")

                        inputs: Dict[str, PortSignature] = {}
                        outputs: Dict[str, PortSignature] = {}

                        if isinstance(cfg, dict) and ("inputs" in cfg or "outputs" in cfg):
                            for p_name, p_val in cfg.get("inputs", {}).items():
                                if isinstance(p_val, dict):
                                    inputs[p_name] = PortSignature(
                                        name=p_name,
                                        signature=AlgebraicSignature(
                                            type_name=str(p_val.get("type_name", in_type or "any")),
                                            state=str(p_val.get("state", in_state or "any")),
                                            qualifiers=frozenset(tuple(q) for q in p_val.get("qualifiers", []))
                                        ),
                                        required=p_val.get("required", True),
                                        default_value=p_val.get("default_value")
                                    )
                            for p_name, p_val in cfg.get("outputs", {}).items():
                                if isinstance(p_val, dict):
                                    outputs[p_name] = PortSignature(
                                        name=p_name,
                                        signature=AlgebraicSignature(
                                            type_name=str(p_val.get("type_name", out_type or "any")),
                                            state=str(p_val.get("state", out_state or "any")),
                                            qualifiers=frozenset(tuple(q) for q in p_val.get("qualifiers", []))
                                        )
                                    )

                        if not inputs:
                            inputs = {"input_data": PortSignature("input_data", in_sig)}
                        if not outputs:
                            outputs = {"output_data": PortSignature("output_data", out_sig)}

                        keywords = set(re.findall(r'[a-zA-Z0-9]+', cell_id.lower()))

                        cell = MicroCell(
                            cell_id=cell_id,
                            stage=stage or 2,
                            keywords=keywords,
                            inputs=inputs,
                            outputs=outputs,
                            domain_name="data_processing",
                            dependencies=deps,
                            code_template=code or "",
                            verified=True
                        )
                        if cell.is_public_morphism:
                            self.loaded_cells[cell.cell_id] = cell

                conn.close()
                logger.info(f"[LATTICE] Loaded {len(self.loaded_cells)} nodes from database.")
            except Exception as e:
                logger.error(f"[LATTICE] Failed loading database: {e}")

    def build_topology(self):
        """
        Builds the lattice directed edges based strictly on monadic type compatibility:
          (u, v) in E <=> u.primary_output.unifies_with(v.primary_input)
        """
        with self._lock:
            self._adjacency.clear()
            self._reverse_adjacency.clear()
            self._cells_by_input.clear()
            self._cells_by_output.clear()

            for cell in self.loaded_cells.values():
                _ = cell.token_set  # Warm up cached token set
                self._adjacency[cell.cell_id] = []
                self._reverse_adjacency[cell.cell_id] = []
                for p in cell.inputs.values():
                    key = (p.type_name, p.state)
                    self._cells_by_input.setdefault(key, []).append(cell)
                for p in cell.outputs.values():
                    key = (p.type_name, p.state)
                    self._cells_by_output.setdefault(key, []).append(cell)

    def get_successors(self, cell: Cell, candidate_pool: Optional[List[Cell]] = None) -> List[Cell]:
        """
        Returns all nodes in candidate_pool whose input can be validly chained from cell's output.
        Computed on-demand via monadic unification check.
        """
        pool = candidate_pool if candidate_pool is not None else self.cells
        out_sig = cell.primary_output
        successors = []
        for cand in pool:
            if cand.cell_id == cell.cell_id:
                continue
            if cand.can_accept(out_sig):
                successors.append(cand)
        return successors

    def get_successors_for_sig(self, sig: AlgebraicSignature) -> List[Cell]:
        """Returns cells whose input unifies with the given output signature."""
        results = []
        seen = set()
        for (in_type, in_state), cells in self._cells_by_input.items():
            cand_sig = AlgebraicSignature(in_type, in_state)
            if sig.unifies_with(cand_sig):
                for c in cells:
                    if c.cell_id not in seen:
                        seen.add(c.cell_id)
                        results.append(c)
        return results
