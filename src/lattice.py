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
import sys
import threading
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, FrozenSet, Union

from log_config import get_logger
try:
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from tokenizer import CellTokenizer

logger = get_logger('lattice')

ABSTRACT_CARRIERS: FrozenSet[str] = frozenset((
    "array-like",
    "array_like",
    "tensor",
    "table",
    "collection",
    "sequence",
    "matrix",
    "scalar",
    "logical",
    "text",
    "path",
    "any",
    "object",
    "*",
    "unknown",
    "top",
    "dataset",
    "numeric",
))

class _UnresolvedPortSentinel:
    """
    Singleton sentinel indicating a port could not be resolved during binding.
    Structurally prevented from being treated as a bound variable or literal value.
    """
    __slots__ = ()
    _instance: Optional["_UnresolvedPortSentinel"] = None

    def __new__(cls) -> "_UnresolvedPortSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNRESOLVED_PORT>"

    def __str__(self) -> str:
        return "<UNRESOLVED_PORT>"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        return other is self or (isinstance(other, str) and other in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>"))

    def __hash__(self) -> int:
        return hash("<UNRESOLVED_PORT>")


UNRESOLVED_PORT = _UnresolvedPortSentinel()


from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Any, Optional, Callable, Set

@dataclass(frozen=True)
class LatticeType:
    """
    Formal Algebraic Free Term for Higher-Kinded Carriers:
      tau ::= B | F[tau_1, ..., tau_n] | Refined(B, phi)
    """
    constructor: str
    parameters: Tuple["LatticeType", ...] = field(default_factory=tuple)
    qualifiers: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_str(cls, s: str) -> "LatticeType":
        if not s:
            return cls(constructor="None")
        s = s.strip()
        quals = ()
        if "{" in s and s.endswith("}"):
            base_part, q_part = s[:-1].split("{", 1)
            s = base_part.strip()
            quals = tuple(q.strip() for q in q_part.split(",") if q.strip())

        if "[" not in s:
            return cls(constructor=s.strip(), qualifiers=quals)

        head, rest = s.split("[", 1)
        inner = rest.rsplit("]", 1)[0]
        parts, depth, cur = [], 0, []
        for ch in inner:
            if ch in "[({":
                depth += 1
            elif ch in "])}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            parts.append("".join(cur).strip())
        return cls(
            constructor=head.strip(),
            parameters=tuple(cls.from_str(p) for p in parts),
            qualifiers=quals
        )

    def is_bottom(self) -> bool:
        return self.constructor.lower() in ("none", "null", "undefined", "bottom", "⊥", "")

    def is_top(self) -> bool:
        return self.constructor.lower() in ("any", "object", "*", "top", "⊤", "unknown")

    def is_subtype_of(self, other: "LatticeType", poset_lookup: Callable[[str, str], bool]) -> bool:
        if self.is_bottom() or other.is_bottom():
            return False
        if other.is_top():
            return True
        if self.is_top() and not other.is_top():
            return False
        if self == other:
            return True

        # Constructor poset subtyping
        if not poset_lookup(self.constructor, other.constructor):
            return False

        # Raw type fallback: F[T1, ..., Tn] <: F
        if len(other.parameters) == 0:
            return True

        # Variadic tuple subtyping: Tuple[S1, ..., Sn] <: Tuple[T, ...]
        if other.constructor.lower() == "tuple" and len(other.parameters) == 2 and other.parameters[1].constructor in ("...", "Ellipsis"):
            elem_t = other.parameters[0]
            if len(self.parameters) == 2 and self.parameters[1].constructor in ("...", "Ellipsis"):
                return self.parameters[0].is_subtype_of(elem_t, poset_lookup)
            if len(self.parameters) > 0:
                return all(p.is_subtype_of(elem_t, poset_lookup) for p in self.parameters)
            return False

        # Sound parameter demand: F </: F[T] if producer is unparameterized
        if len(self.parameters) == 0 and len(other.parameters) > 0:
            return all(p.is_top() for p in other.parameters)

        if len(self.parameters) != len(other.parameters):
            return False

        if other.qualifiers:
            if not set(other.qualifiers).issubset(set(self.qualifiers)):
                return False

        return all(p1.is_subtype_of(p2, poset_lookup) for p1, p2 in zip(self.parameters, other.parameters))

    def __str__(self) -> str:
        s = self.constructor
        if self.parameters:
            s += f"[{', '.join(str(p) for p in self.parameters)}]"
        if self.qualifiers:
            s += f"{{{', '.join(self.qualifiers)}}}"
        return s

class TypeRegistry:
    """
    Dynamic Poset Type Hierarchy (T, <=).
    Maintains a directed acyclic graph of subtype relationships.
    Types are registered dynamically from trees without hardcoded domain dependencies.
    Subtyping is verified dynamically using Python's MRO, standard protocols, and poset reachability.
    """
    _instance: Optional[TypeRegistry] = None
    _lock = threading.RLock()

    def __init__(self):
        self._parents: Dict[str, Set[str]] = {}
        self._aliases: Dict[str, str] = {}
        self._state_parents: Dict[str, str] = {}
        self._bootstrap_carrier_hierarchy()
        self._bootstrap_state_hierarchy()
        self._bootstrap_plugin_types()

    def _bootstrap_carrier_hierarchy(self):
        """Initializes universal, language-agnostic computational carrier types."""
        # Textual / path carriers
        for t in ("filepath", "filename", "pathname", "pathlike"):
            self.register_type(t, "path")
        for t in ("path", "uri", "url", "text"):
            self.register_type(t, "str")
        # Numeric carriers
        for t in ("int", "float", "complex"):
            self.register_type(t, "numeric")
        self.register_type("numeric", "scalar")
        self.register_type("bool", "logical")
        # Functor and Container Constructors
        for ctor in ("list", "sequence", "collection", "tuple", "set", "dict", "file"):
            self.register_type(ctor, "object")
        self.register_type("file", "pathlike")
        self.register_type("file", "filepath")
        self.register_type("file", "path")
        self.register_type("file", "str")

        # Modality and Format Base Taxonomy
        self.register_type("modality", "object")
        self.register_type("format", "object")
        self.register_type("imageformat", "format")
        self.register_type("tabularformat", "format")
        self.register_type("arrayformat", "format")
        self.register_type("modelformat", "format")
        self.register_type("canvas", "object")

        # Collection carriers
        for t in ("list", "tuple", "set", "frozenset", "dict", "mapping", "map"):
            self.register_type(t, "collection")
        self.register_type("sequence", "collection")
        self.register_type("list", "sequence")
        self.register_type("tuple", "sequence")
        # Tensor / array carriers
        self.register_type("ndarray", "array-like")
        self.register_type("matrix", "array-like")
        self.register_type("dataframe", "array-like")
        self.register_type("series", "array-like")
        self.register_type("list", "array-like")
        self.register_type("tuple", "array-like")
        self.register_type("array-like", "tensor")
        self.register_type("array-like", "sequence")
        self.register_type("buffer", "tensor")
        # Tabular carriers
        for t in ("dataframe", "series", "dataset"):
            self.register_type(t, "table")
            self.register_type(t, "array-like")
        # Aliases
        for alias, can in (
            ("string", "str"),
            ("boolean", "bool"),
            ("integer", "int"),
            ("dictionary", "dict"),
            ("number", "numeric"),
            ("array_like", "array-like"),
            ("matlike", "tensor"),
        ):
            self.register_alias(alias, can)

    def _bootstrap_state_hierarchy(self):
        """Initializes canonical, universal typestate preorder poset (S, <=)."""
        # Machine learning features, partitions, and target states
        self.register_state("split_train_features", "unscaled_features")
        self.register_state("split_test_features", "unscaled_features")
        self.register_state("unscaled_features", "feature_matrix")
        self.register_state("scaled_features", "feature_matrix")
        self.register_state("imputed_features", "unscaled_features")
        self.register_state("encoded_features", "unscaled_features")
        self.register_state("reduced_features", "feature_matrix")
        self.register_state("feature_matrix", "ndarray_generic")

        self.register_state("split_train_targets", "target_vector")
        self.register_state("split_test_targets", "target_vector")
        self.register_state("target_vector", "ndarray_generic")

        # Tabular states
        self.register_state("series_cleaned", "series_numeric")
        self.register_state("series_numeric", "series_raw")
        self.register_state("dataframe_2d_generic", "raw_dataset")
        self.register_state("cleaned", "raw_dataset")
        self.register_state("normalized", "cleaned")
        self.register_state("transformed", "cleaned")
        self.register_state("deduped", "cleaned")
        self.register_state("filtered", "cleaned")
        self.register_state("indexed", "cleaned")
        self.register_state("numeric_only", "cleaned")

        # Estimator states
        self.register_state("fit_regressor", "fit_estimator")
        self.register_state("fit_classifier", "fit_estimator")
        self.register_state("fit_clusterer", "fit_estimator")
        self.register_state("fit_transformer", "fit_estimator")
        self.register_state("fit_estimator", "trained")
        self.register_state("unfit_estimator", "default")

        # Computer vision states
        self.register_state("binary", "gray")
        self.register_state("grayscale", "gray")
        self.register_state("computed_threshold", "binary")
        self.register_state("blurred", "gray")
        self.register_state("edge_map", "gray")
        self.register_state("dilated", "binary")
        self.register_state("eroded", "binary")
        self.register_state("morphology_processed", "binary")
        self.register_state("contours", "collection")

        # Egress / storage states
        self.register_state("saved", "written_to_disk")
        self.register_state("figure_saved", "written_to_disk")
        self.register_state("saved_npy", "written_to_disk")
        self.register_state("saved_npz", "written_to_disk")
        self.register_state("written_to_disk", "exported")

        # Path and source identifier states
        self.register_state("valid_path", "source_identifier")
        self.register_state("file_path", "source_identifier")
        self.register_state("file_path_str", "source_identifier")
        self.register_state("source_path", "source_identifier")

    def _bootstrap_plugin_types(self):
        """Dynamically ingests domain plugin types from trees/*.json without engine hardcoding."""
        import glob, json
        for s_dir in ("trees", "new trees"):
            if not os.path.exists(s_dir):
                continue
            for p in sorted(glob.glob(f"{s_dir}/*.json")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "types" in data and isinstance(data["types"], dict):
                        for t_name, t_meta in data["types"].items():
                            parent = t_meta.get("parent") if isinstance(t_meta, dict) else str(t_meta)
                            if parent:
                                self.register_type(t_name, parent)
                except Exception:
                    pass

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

    def register_type(self, type_name: str, super_type: Optional[str] = None):
        """Registers a type and optionally declares its supertype in the poset."""
        name = str(type_name).strip().lower()
        if not name or name in ("none", "null"):
            return
        if name not in self._parents:
            self._parents[name] = set()
        if super_type:
            super_name = str(super_type).strip().lower()
            if super_name and super_name != name and super_name not in ("none", "null"):
                # Poset invariant: Distinct top-level carriers never subsume each other
                top_level_carriers = {"scalar", "collection", "tensor", "table", "logical", "str"}
                if name in top_level_carriers and super_name in top_level_carriers:
                    return
                # Prevent cycles in the poset
                if self.is_subtype(super_name, name):
                    return
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

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _resolve_runtime_class(type_name: str) -> Optional[type]:
        """Dynamically locates a runtime class via Python introspection without hardcoding."""
        if not type_name or not isinstance(type_name, str):
            return None
        import builtins
        import numbers
        import warnings
        clean = type_name.strip()
        clean_lower = clean.lower()
        if clean_lower in ("numeric", "number"):
            return numbers.Number
        if hasattr(builtins, clean):
            obj = getattr(builtins, clean)
            if isinstance(obj, type):
                return obj
        if hasattr(builtins, clean_lower):
            obj = getattr(builtins, clean_lower)
            if isinstance(obj, type):
                return obj
        if "." in clean:
            parts = clean.rsplit(".", 1)
            mod_name, cls_name = parts[0], parts[1]
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, cls_name):
                obj = getattr(mod, cls_name)
                if isinstance(obj, type):
                    return obj
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mod_name, mod in list(sys.modules.items()):
                if not mod or mod_name.startswith("_") or "._" in mod_name:
                    continue
                try:
                    if hasattr(mod, clean):
                        obj = getattr(mod, clean)
                        if isinstance(obj, type):
                            return obj
                except Exception:
                    continue
        return None

    @functools.lru_cache(maxsize=16384)
    def is_subtype(self, sub: str, super_: str) -> bool:
        """
        Computes poset partial order: returns True iff sub <= super_.
        Supports recursive structural generic subtyping for terms F[tau_1, ..., tau_n].
        Wildcards ('any', 'object', '*', 'top') are Top types that subsume all types.
        Uses formal MRO and poset reachability dynamically.
        """
        if not sub or not super_:
            return False
        sub_str = str(sub).strip()
        super_str = str(super_).strip()

        if sub_str.lower() in ("none", "null", "undefined", "") or super_str.lower() in ("none", "null", "undefined", ""):
            return False

        if "[" in sub_str or "[" in super_str or "{" in sub_str or "{" in super_str:
            t1 = LatticeType.from_str(sub_str)
            t2 = LatticeType.from_str(super_str)
            return t1.is_subtype_of(t2, self._atomic_is_subtype)

        return self._atomic_is_subtype(sub_str, super_str)

    def _atomic_is_subtype(self, sub: str, super_: str) -> bool:
        if not sub or not super_:
            return False
        sub_c = self.canonical_name(sub)
        super_c = self.canonical_name(super_)

        sub_l = sub_c.lower()
        super_l = super_c.lower()

        if sub_l in ("none", "null", "undefined", "") or super_l in ("none", "null", "undefined", ""):
            return False

        if super_l in ("any", "object", "*", "top", "unknown"):
            return True
        if sub_l in ("any", "top", "*"):
            return False
        if sub_l == super_l:
            return True

        if "_or_" in sub_l:
            if any(self._atomic_is_subtype(part, super_c) for part in sub_l.split("_or_")):
                return True

        sub_key = sub_l.rsplit(".", 1)[-1] if sub_l not in self._parents and "." in sub_l else sub_l
        super_key = super_l.rsplit(".", 1)[-1] if super_l not in self._parents and "." in super_l else super_l
        if sub_key in self._parents:
            visited = set()
            from collections import deque
            queue = deque([sub_key])
            while queue:
                curr = queue.popleft()
                if curr == super_key or curr == super_l:
                    return True
                visited.add(curr)
                for parent in self._parents.get(curr, []):
                    p_l = parent.lower()
                    if p_l not in visited:
                        queue.append(p_l)

        sub_cls = self._resolve_runtime_class(sub_c)
        super_cls = self._resolve_runtime_class(super_c)
        if sub_cls is not None and super_cls is not None:
            try:
                if issubclass(sub_cls, super_cls):
                    return True
            except TypeError:
                pass

        return False
    def register_state(self, state: str, parent_state: Optional[str] = None):
        """Registers a typestate and its optional parent_state."""
        s = str(state).strip().lower()
        if not s:
            return
        if parent_state:
            p = str(parent_state).strip().lower()
            if p and p != s:
                self._state_parents[s] = p

    def get_state_parent(self, state: str) -> Optional[str]:
        """Returns the declared parent_state of a typestate, if any."""
        return self._state_parents.get(str(state).strip().lower())

    def is_state_compatible(
        self,
        producer_state: str,
        consumer_state: str,
        producer_accepted: Union[Set[str], FrozenSet[str], List[str]] = frozenset(),
        consumer_accepted: Union[Set[str], FrozenSet[str], List[str]] = frozenset(),
        consumer_parent: Optional[str] = None,
        producer_parent: Optional[str] = None,
    ) -> bool:
        """
        Evaluates typestate compatibility between producer output and consumer input.
        Rules:
          1. Wildcard match if either state is 'any' or '*'.
          2. Exact match (case-insensitive).
          3. Accept if consumer_state is in producer's accepted_states (or producer_state in consumer's accepted_states).
          4. Walk parent_state up consumer's declared chain before rejecting.
          5. Walk parent_state up producer's declared chain before rejecting.
        """
        p_state = str(producer_state or "any").strip().lower()
        c_state = str(consumer_state or "any").strip().lower()

        if p_state in ("any", "*") or c_state in ("any", "*"):
            return True

        if p_state == c_state:
            return True

        p_acc = {str(s).strip().lower() for s in (producer_accepted or []) if str(s).strip()}
        c_acc = {str(s).strip().lower() for s in (consumer_accepted or []) if str(s).strip()}

        # (a) Accept if other_sig.state is in producer's accepted_states (or producer_state in consumer's accepted_states)
        if c_state in p_acc or p_state in c_acc:
            return True

        # (b) Walk parent_state up consumer's declared chain before rejecting
        visited_c = set()
        curr_c = str(consumer_parent).strip().lower() if consumer_parent else self.get_state_parent(c_state)
        while curr_c and curr_c not in visited_c:
            visited_c.add(curr_c)
            if curr_c == p_state or curr_c in p_acc:
                return True
            curr_c = self.get_state_parent(curr_c)

        # Also walk parent_state up producer's declared chain
        visited_p = set()
        curr_p = str(producer_parent).strip().lower() if producer_parent else self.get_state_parent(p_state)
        while curr_p and curr_p not in visited_p:
            visited_p.add(curr_p)
            if curr_p == c_state or curr_p in c_acc:
                return True
            curr_p = self.get_state_parent(curr_p)

        return False


def is_subtype(sub: str, parent: str) -> bool:
    return TypeRegistry.get_instance().is_subtype(sub, parent)


def canonical_type_name(type_name: str) -> str:
    return TypeRegistry.get_instance().canonical_name(type_name)


def _clean_abs_carrier(val: Any) -> str:
    if not val:
        return ""
    s = str(val).strip()
    if s.lower() in ("none", "null", ""):
        return ""
    return s


def _normalize_qualifiers(raw_qualifiers: Any) -> FrozenSet[Tuple[str, ...]]:
    if not raw_qualifiers:
        return frozenset()
    result = []
    for q in raw_qualifiers:
        if isinstance(q, (list, tuple)):
            result.append(tuple(str(item) for item in q))
        elif isinstance(q, str):
            result.append((q,))
        else:
            result.append((str(q),))
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class AlgebraicSignature:
    """
    Formal typestate signature: tau = (type_name, state, qualifiers, abstract_type, accepted_states, parent_state).
    Conforms to Section 3.1 of the NSTL paper.
    """
    type_name: str = "any"
    state: str = "any"
    qualifiers: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)
    abstract_type: str = ""
    accepted_states: FrozenSet[str] = field(default_factory=frozenset)
    parent_state: Optional[str] = None

    def __post_init__(self):
        cleaned = _clean_abs_carrier(self.abstract_type)
        if cleaned != self.abstract_type:
            object.__setattr__(self, "abstract_type", cleaned)
        if self.accepted_states and not isinstance(self.accepted_states, frozenset):
            object.__setattr__(self, "accepted_states", frozenset(str(s).strip().lower() for s in self.accepted_states if str(s).strip()))
        if self.parent_state:
            object.__setattr__(self, "parent_state", str(self.parent_state).strip().lower())
        if self.qualifiers is not None:
            norm_q = _normalize_qualifiers(self.qualifiers)
            if norm_q != self.qualifiers:
                object.__setattr__(self, "qualifiers", norm_q)

    @classmethod
    def from_string(cls, type_name: str, state: str = "any", abstract_type: str = "", accepted_states: Optional[Any] = None, parent_state: Optional[str] = None) -> "AlgebraicSignature":
        acc = frozenset(str(s).strip().lower() for s in (accepted_states or []) if str(s).strip()) if accepted_states else frozenset()
        p = str(parent_state).strip().lower() if parent_state else None
        return cls(type_name=type_name, state=state, abstract_type=_clean_abs_carrier(abstract_type), accepted_states=acc, parent_state=p)

    def is_top(self) -> bool:
        canonical = TypeRegistry.get_instance().canonical_name(self.type_name)
        return canonical.lower() in ("any", "*", "top", "object", "unknown")

    def unifies_with(self, other: Any) -> bool:
        """
        Evaluates whether producer output `self` can satisfy consumer input `other`.
        Rules:
          1. Consumer Top accepts any producer type.
          2. Non-top consumer rejects top producer.
          3. State compatibility check:
             - matches if either is 'any' or '*'
             - matches if exact match
             - (a) accepts if other_sig.state is in producer's accepted_states (or vice versa)
             - (b) walks parent_state up consumer's declared chain before rejecting
          4. Producer type must be a subtype of consumer type in the Type Poset,
             or unify via compatible abstract categorical carriers.
          5. Consumer qualifiers must be a subset of producer qualifiers.
        """
        if hasattr(other, "signature") and isinstance(other.signature, AlgebraicSignature):
            other_sig = other.signature
        elif isinstance(other, AlgebraicSignature):
            other_sig = other
        else:
            return False

        # State compatibility check
        registry = TypeRegistry.get_instance()
        if not registry.is_state_compatible(
            producer_state=self.state,
            consumer_state=other_sig.state,
            producer_accepted=self.accepted_states,
            consumer_accepted=other_sig.accepted_states,
            consumer_parent=other_sig.parent_state,
            producer_parent=self.parent_state,
        ):
            return False

        # Consumer accepts anything
        if other_sig.is_top():
            return True
        # Producer is untyped wildcard, cannot guarantee concrete type requirement
        if self.is_top():
            return False

        # Category-Theoretic Generic Functor & Type Variable Unification
        p_tn = (self.type_name or "").strip()
        c_tn = (other_sig.type_name or "").strip()
        is_generic_match = False
        
        # 1. Bare type variables (T, U, V, State, Comparable, C, R) universally unify
        if c_tn in ("T", "U", "V", "State", "Comparable", "C", "R") or p_tn in ("T", "U", "V", "State", "Comparable", "C", "R"):
            is_generic_match = True
        else:
            # 2. Parametric Container / Functor unification (e.g. List[Contour] <-> List[T])
            import re
            m_c = re.match(r"^(\w+)\[(.*)\]$", c_tn)
            m_p = re.match(r"^(\w+)\[(.*)\]$", p_tn)
            if m_c and m_p:
                c_ctor, c_inner = m_c.group(1).lower(), m_c.group(2).strip()
                p_ctor, p_inner = m_p.group(1).lower(), m_p.group(2).strip()
                if c_ctor == p_ctor:
                    if (
                        c_inner in ("T", "U", "V", "State", "Comparable", "C", "R", "Any", "*")
                        or p_inner in ("T", "U", "V", "State", "Comparable", "C", "R", "Any", "*")
                    ):
                        is_generic_match = True
            elif (c_tn.startswith("List[") and p_tn.lower() in ("list", "sequence", "iterable", "collection")) or                  (p_tn.startswith("List[") and c_tn.lower() in ("list", "sequence", "iterable", "collection")):
                is_generic_match = True

        # Poset subtyping
        registry = TypeRegistry.get_instance()
        if not is_generic_match and not registry.is_subtype(self.type_name, other_sig.type_name):
            # Poset abstract carrier compatibility fallback:
            # Fallback ONLY applies when consumer target is expecting an abstract carrier/interface,
            # NOT when consumer target is a concrete class receiver (e.g. MultiIndex, LinearRegression).
            other_tn = (other_sig.type_name or "").strip().lower()
            other_abs = (other_sig.abstract_type or "").strip().lower()
            is_consumer_abstract = (other_tn in ABSTRACT_CARRIERS or other_tn == other_abs)
            if (
                is_consumer_abstract
                and self.abstract_type
                and other_sig.abstract_type
                and registry.is_subtype(self.abstract_type, other_sig.abstract_type)
            ):
                pass
            else:
                return False

        # Qualifier satisfaction
        if other_sig.qualifiers and not other_sig.qualifiers.issubset(self.qualifiers):
            ignorable = {("const",), ("scalar",), ("vector",), ("matrix",), ("primary",)}
            req = {q for q in other_sig.qualifiers if q not in ignorable}
            if req and not req.issubset(self.qualifiers):
                return False

        return True

    def matches(self, other: Any) -> bool:
        return self.unifies_with(other)


class CaseInsensitiveDict(dict):
    """
    Case-insensitive dictionary for cell lookups across lowercase/uppercase identifiers.
    Preserves exact keys while allowing case-insensitive retrieval.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lower_map: Dict[str, Any] = {str(k).lower(): k for k in self.keys()}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._lower_map[str(key).lower()] = key

    def __delitem__(self, key):
        super().__delitem__(key)
        self._lower_map.pop(str(key).lower(), None)

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        lower = str(key).lower()
        if lower in self._lower_map:
            return super().__getitem__(self._lower_map[lower])
        raise KeyError(key)

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return str(key).lower() in self._lower_map

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key, *args):
        if super().__contains__(key):
            self._lower_map.pop(str(key).lower(), None)
            return super().pop(key, *args)
        lower = str(key).lower()
        if lower in self._lower_map:
            actual = self._lower_map.pop(lower)
            return super().pop(actual, *args)
        if args:
            return args[0]
        raise KeyError(key)

    def clear(self):
        super().clear()
        self._lower_map.clear()


class PortMapping(dict):
    """
    Port dictionary supporting canonical port names as primary keys,
    while permitting alias access (e.g. 'port_0', 'port_1') for backward compatibility.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases: Dict[str, str] = {}

    def add_alias(self, alias: str, canonical: str):
        self._aliases[alias] = canonical

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self._aliases:
            return super().__getitem__(self._aliases[key])
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return super().__contains__(key) or key in self._aliases


class PortSignature:
    """Named port carrying an AlgebraicSignature typestate."""
    __slots__ = [
        "name", "signature", "required", "default_value", "doc", "domain",
        "abstract_type", "enum_values", "param_kind", "value_constraints", "shape_contract",
        "accepted_states", "parent_state", "port_role"
    ]

    def __init__(
        self,
        name: str = "",
        signature: Union[AlgebraicSignature, str, Any] = "any",
        required: bool = True,
        default_value: Optional[Any] = None,
        doc: str = "",
        domain: str = "",
        abstract_type: str = "",
        enum_values: Optional[List[Any]] = None,
        param_kind: str = "standard",
        value_constraints: Optional[Dict[str, Any]] = None,
        shape_contract: Optional[Dict[str, Any]] = None,
        accepted_states: Optional[Union[List[str], Set[str], FrozenSet[str]]] = None,
        parent_state: Optional[str] = None,
        port_role: Optional[str] = None,
        **kwargs
    ):
        self.name = str(name)
        self.domain = str(domain or kwargs.get("domain", ""))
        self.abstract_type = _clean_abs_carrier(abstract_type or kwargs.get("abstract_type", ""))
        self.enum_values = enum_values if enum_values is not None else kwargs.get("enum_values")
        self.param_kind = str(param_kind or kwargs.get("param_kind", "standard"))
        self.value_constraints = value_constraints if value_constraints is not None else kwargs.get("value_constraints")
        self.shape_contract = shape_contract if shape_contract is not None else kwargs.get("shape_contract")
        self.port_role = str(port_role).strip().lower() if port_role else (str(kwargs.get("port_role")).strip().lower() if kwargs.get("port_role") else None)

        acc = accepted_states if accepted_states is not None else kwargs.get("accepted_states", [])
        if isinstance(acc, (set, frozenset, list, tuple)):
            self.accepted_states = frozenset(str(s).strip().lower() for s in acc if str(s).strip())
        else:
            self.accepted_states = frozenset()

        raw_parent = parent_state or kwargs.get("parent_state")
        self.parent_state = str(raw_parent).strip().lower() if raw_parent else None

        if isinstance(signature, AlgebraicSignature):
            acc_combined = self.accepted_states or signature.accepted_states
            p_combined = self.parent_state or signature.parent_state
            if (self.abstract_type and not signature.abstract_type) or (acc_combined != signature.accepted_states) or (p_combined != signature.parent_state):
                self.signature = AlgebraicSignature(
                    type_name=signature.type_name,
                    state=signature.state,
                    qualifiers=signature.qualifiers,
                    abstract_type=self.abstract_type or signature.abstract_type,
                    accepted_states=acc_combined,
                    parent_state=p_combined
                )
            else:
                self.signature = signature
            if signature.abstract_type:
                self.abstract_type = signature.abstract_type
            if signature.accepted_states and not self.accepted_states:
                self.accepted_states = signature.accepted_states
            if signature.parent_state and not self.parent_state:
                self.parent_state = signature.parent_state
        elif hasattr(signature, "signature") and isinstance(signature.signature, AlgebraicSignature):
            sig = signature.signature
            self.signature = AlgebraicSignature(
                type_name=sig.type_name,
                state=sig.state,
                qualifiers=sig.qualifiers,
                abstract_type=self.abstract_type or sig.abstract_type,
                accepted_states=self.accepted_states or sig.accepted_states,
                parent_state=self.parent_state or sig.parent_state
            )
            if self.signature.abstract_type:
                self.abstract_type = self.signature.abstract_type
        elif isinstance(signature, str):
            if "name" in kwargs:
                self.name = kwargs["name"]
                self.signature = AlgebraicSignature(
                    type_name=signature,
                    state=kwargs.get("state", "any"),
                    abstract_type=self.abstract_type,
                    accepted_states=self.accepted_states,
                    parent_state=self.parent_state
                )
            elif signature != "any":
                self.name = kwargs.get("name", f"port_{name}")
                self.signature = AlgebraicSignature(
                    type_name=name,
                    state=signature,
                    abstract_type=self.abstract_type,
                    accepted_states=self.accepted_states,
                    parent_state=self.parent_state
                )
            else:
                self.signature = AlgebraicSignature(
                    type_name=name if name else "any",
                    state="any",
                    abstract_type=self.abstract_type,
                    accepted_states=self.accepted_states,
                    parent_state=self.parent_state
                )
        else:
            self.signature = AlgebraicSignature(
                "any", "any",
                abstract_type=self.abstract_type,
                accepted_states=self.accepted_states,
                parent_state=self.parent_state
            )

        self.required = bool(required)
        self.default_value = default_value
        self.doc = str(doc or "")

    @property
    def type_name(self) -> str:
        return self.signature.type_name

    @property
    def state(self) -> str:
        return self.signature.state

    @property
    def description(self) -> str:
        return self.doc or ""

    @property
    def derived_role(self) -> str:
        if self.port_role:
            return self.port_role

        name_lower = self.name.lower()
        state_lower = str(getattr(self.signature, "state", "")).lower()
        desc_lower = (self.doc or "").lower()
        qualifiers = {q[0].lower() for q in getattr(self.signature, "qualifiers", []) if q}

        # Check for feature input
        if (
            "feature" in state_lower
            or "feature" in name_lower
            or "matrix" in qualifiers
            or (self.name in ("X", "x") or self.name.startswith(("X_", "x_")))
            or "matrix" in desc_lower
        ):
            return "feature_input"

        # Check for target input
        if (
            "target" in state_lower
            or "label" in state_lower
            or "target" in name_lower
            or "label" in name_lower
            or "vector" in qualifiers
            or (self.name == "y" or self.name.startswith("y_"))
            or "vector" in desc_lower
        ):
            return "target_input"

        # Check for model / estimator input
        if (
            "model" in name_lower
            or "estimator" in name_lower
            or "model" in state_lower
            or "estimator" in state_lower
        ):
            return "model_input"

        # Check for tabular / structural data input
        t_name = self.type_name.lower()
        if (
            t_name in ("dataframe", "table", "dataset")
            or "dataframe" in state_lower
            or "adjacency" in state_lower
            or "graph" in state_lower
            or "data" in state_lower
            or name_lower in ("df", "dataframe", "data", "graph", "dataset")
        ):
            return "data_input"

        # Check for source data / file input
        if (
            "source" in state_lower
            or "source" in name_lower
            or "filepath" in name_lower
            or name_lower in ("file_path", "filename", "path")
            or "source_identifier" in state_lower
        ):
            return "source_data"

        # Check for model / destination sink
        if (
            "sink" in state_lower
            or "dest" in state_lower
            or "target_path" in name_lower
            or "output_path" in name_lower
            or "savepath" in name_lower
            or "dest_identifier" in state_lower
        ):
            return "model_sink"

        return "standard"

    def is_top(self) -> bool:
        return self.signature.is_top()

    def unifies_with(self, other: Any) -> bool:
        if isinstance(other, PortSignature):
            return self.signature.unifies_with(other.signature)
        if isinstance(other, AlgebraicSignature):
            return self.signature.unifies_with(other)
        return False

    def __repr__(self) -> str:
        abs_info = f", abs={self.abstract_type}" if self.abstract_type else ""
        acc_info = f", acc={list(self.accepted_states)}" if self.accepted_states else ""
        return f"Port({self.name}: {self.signature.type_name}[{self.signature.state}]{abs_info}{acc_info}, req={self.required})"


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
        "source_priority", "source_provenance", "is_public",
        "topology_type", "feedback_state_type", "bound_slots",
        "replica_of", "replica_role",
        "mutation_type", "is_context_manager", "raises", "type_vars",
        "preconditions", "postconditions", "effects", "edges", "endable",
        "_primary_input", "_primary_output", "_token_set", "_token_count",
        "_identity_tokens"
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
        source_provenance: Optional[str] = "unknown",
        topology_type: str = "sequential",
        feedback_state_type: Optional[str] = None,
        bound_slots: Optional[Dict[str, Any]] = None,
        is_public: bool = True,
        mutation_type: str = "pure",
        is_context_manager: bool = False,
        raises: Optional[List[str]] = None,
        type_vars: Optional[List[str]] = None,
        preconditions: Optional[List[Any]] = None,
        postconditions: Optional[List[Any]] = None,
        effects: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        endable: Optional[bool] = None,
        **kwargs
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = set(str(k).lower() for k in keywords if len(str(k)) >= 3) if keywords else set()
        self.cell_type = cell_type
        self.domain_name = domain_name
        self.node_type = node_type
        self.node_role = str(node_role).lower() if node_role else "function"
        if isinstance(slots, (list, set, tuple)):
            self.slots = {s: {} for s in slots}
        else:
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
        self.is_public = bool(is_public)
        self.mutation_type = str(mutation_type or kwargs.get("mutation_type", "pure"))
        self.is_context_manager = bool(is_context_manager or kwargs.get("is_context_manager", False))
        self.raises = list(raises or kwargs.get("raises", []))
        self.type_vars = list(type_vars or kwargs.get("type_vars", []))

        # Infer topology type if default sequential but node specifies control flow
        nt = str(self.node_type).lower()
        if topology_type == "sequential":
            if nt.startswith("macro_loop") or "loop" in nt or feedback_state_type:
                topology_type = "traced_loop"
            elif nt.startswith("macro_conditional") or "conditional" in nt:
                topology_type = "coproduct_branch"
            elif nt.startswith("macro_operator") or "monoidal" in nt:
                topology_type = "monoidal_product"

        self.topology_type = topology_type
        self.feedback_state_type = feedback_state_type
        self.bound_slots = dict(bound_slots) if bound_slots else {}

        self.preconditions = list(preconditions or kwargs.get("preconditions", []))
        eff = effects if effects is not None else postconditions
        if eff is None:
            eff = kwargs.get("effects", kwargs.get("postconditions", []))
        self.postconditions = list(eff or [])
        self.effects = self.postconditions
        self.edges = list(edges or kwargs.get("edges", []))
        self.endable = endable if endable is not None else kwargs.get("endable")

        # For-each multiplicity expansion: a replica is a runtime copy of a
        # planned cell that re-consumes its receiver from the environment and
        # binds the NEXT member of an identifier role group (X, Y, Z). None on
        # original cells.
        self.replica_of: Optional[str] = None
        self.replica_role: Optional[str] = None

        self._primary_input = None
        self._primary_output = None

        # Extract declared slot names from slots or code_template to resolve generic 'port_X' keys
        slot_names_list = []
        if isinstance(slots, (list, tuple, set)):
            slot_names_list = list(slots)
        elif isinstance(slots, dict):
            slot_names_list = list(slots.keys())

        m_assign = re.match(r'^\s*(\{.+?\})\s*=\s*(.+)$', code_template)
        if m_assign:
            lhs_matches = re.findall(r'\{([a-zA-Z0-9_]+)\}', m_assign.group(1))
            lhs_outs = set(lhs_matches)
        else:
            lhs_matches = []
            lhs_outs = set()
        lhs_outs.add("output_var")
        in_slots_derived = [s for s in slot_names_list if s not in lhs_outs]
        if not in_slots_derived and code_template:
            rhs = m_assign.group(2) if m_assign else code_template
            rhs_matches = re.findall(r'\{([a-zA-Z0-9_]+)\}', rhs)
            seen_m = set()
            in_slots_derived = [m for m in rhs_matches if not (m in seen_m or seen_m.add(m))]

        # Normalize inputs into Dict[str, PortSignature]
        if isinstance(inputs, list):
            inputs_dict = {}
            for idx, item in enumerate(inputs):
                if isinstance(item, dict):
                    p_name = item.get("name") or item.get("port_name") or f"port_{idx}"
                    inputs_dict[p_name] = item
                elif isinstance(item, (PortSignature, AlgebraicSignature)):
                    inputs_dict[getattr(item, "name", f"port_{idx}")] = item
                else:
                    inputs_dict[f"port_{idx}"] = item
            inputs = inputs_dict

        self.inputs: Dict[str, PortSignature] = PortMapping()
        for k, v in (inputs or {}).items():
            orig_k = k
            if k.startswith("port_"):
                try:
                    p_idx = int(k.split("_")[1])
                    if p_idx < len(in_slots_derived):
                        k = in_slots_derived[p_idx]
                except (ValueError, IndexError):
                    pass

            if isinstance(v, PortSignature):
                port_sig = v
            elif isinstance(v, AlgebraicSignature):
                port_sig = PortSignature(name=k, signature=v, accepted_states=v.accepted_states, parent_state=v.parent_state)
            elif isinstance(v, dict):
                abs_t = _clean_abs_carrier(v.get("abstract_type"))
                acc_s = frozenset(str(s).strip().lower() for s in (v.get("accepted_states") or []) if str(s).strip())
                p_s = str(v.get("parent_state") or "").strip().lower() or None
                sig = AlgebraicSignature(
                    type_name=v.get("type_name", "any"),
                    state=v.get("state", "any"),
                    qualifiers=_normalize_qualifiers(v.get("qualifiers", [])),
                    abstract_type=abs_t,
                    accepted_states=acc_s,
                    parent_state=p_s,
                )
                port_sig = PortSignature(
                    name=k,
                    signature=sig,
                    required=v.get("required", True),
                    default_value=v.get("default_value"),
                    doc=v.get("doc", v.get("description", "")),
                    domain=v.get("domain", ""),
                    abstract_type=abs_t,
                    enum_values=v.get("enum_values"),
                    param_kind=v.get("param_kind", "standard"),
                    value_constraints=v.get("value_constraints"),
                    shape_contract=v.get("shape_contract"),
                    accepted_states=acc_s,
                    parent_state=p_s,
                    port_role=v.get("port_role"),
                )
            else:
                port_sig = PortSignature(name=k, signature=AlgebraicSignature("any", "any"))

            port_sig.name = k
            self.inputs[k] = port_sig
            if orig_k != k:
                self.inputs.add_alias(orig_k, k)

        # Normalize outputs into Dict[str, PortSignature]
        if isinstance(outputs, list):
            outputs_dict = {}
            for idx, item in enumerate(outputs):
                if isinstance(item, dict):
                    p_name = item.get("name") or item.get("port_name") or f"port_{idx}"
                    outputs_dict[p_name] = item
                elif isinstance(item, (PortSignature, AlgebraicSignature)):
                    outputs_dict[getattr(item, "name", f"port_{idx}")] = item
                else:
                    outputs_dict[f"port_{idx}"] = item
            outputs = outputs_dict

        self.outputs: Dict[str, PortSignature] = PortMapping()
        for k, v in (outputs or {}).items():
            orig_k = k
            if k.startswith("port_"):
                try:
                    p_idx = int(k.split("_")[1])
                    if p_idx < len(lhs_matches):
                        k = lhs_matches[p_idx]
                    elif len(lhs_matches) == 1:
                        k = lhs_matches[0]
                except (ValueError, IndexError):
                    pass

            if isinstance(v, PortSignature):
                v.name = k
                self.outputs[k] = v
            elif isinstance(v, AlgebraicSignature):
                self.outputs[k] = PortSignature(name=k, signature=v, accepted_states=v.accepted_states, parent_state=v.parent_state)
            elif isinstance(v, dict):
                abs_t = _clean_abs_carrier(v.get("abstract_type"))
                acc_s = frozenset(str(s).strip().lower() for s in (v.get("accepted_states") or []) if str(s).strip())
                p_s = str(v.get("parent_state") or "").strip().lower() or None
                sig = AlgebraicSignature(
                    type_name=v.get("type_name", "any"),
                    state=v.get("state", "any"),
                    qualifiers=_normalize_qualifiers(v.get("qualifiers", [])),
                    abstract_type=abs_t,
                    accepted_states=acc_s,
                    parent_state=p_s,
                )
                self.outputs[k] = PortSignature(
                    name=k,
                    signature=sig,
                    required=v.get("required", True),
                    default_value=v.get("default_value"),
                    doc=v.get("doc", v.get("description", "")),
                    domain=v.get("domain", ""),
                    abstract_type=abs_t,
                    enum_values=v.get("enum_values"),
                    param_kind=v.get("param_kind", "standard"),
                    value_constraints=v.get("value_constraints"),
                    shape_contract=v.get("shape_contract"),
                    accepted_states=acc_s,
                    parent_state=p_s,
                    port_role=v.get("port_role"),
                )
            else:
                self.outputs[k] = PortSignature(name=k, signature=AlgebraicSignature("any", "any"))

            if orig_k != k:
                self.outputs.add_alias(orig_k, k)

        self._token_set: Optional[Set[str]] = None
        self._token_count: int = 0
        self._identity_tokens: Optional[Set[str]] = None

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
            if self.docstring:
                toks.update(CellTokenizer.tokenize_prompt(self.docstring))
            for p in self.inputs:
                toks.update(CellTokenizer.tokenize_identifier(p))
            for p in self.outputs:
                toks.update(CellTokenizer.tokenize_identifier(p))
            self._token_set = toks
            self._token_count = len(toks)
        return self._token_set

    @property
    def identity_tokens(self) -> Set[str]:
        """
        The cell's IDENTITY vocabulary: tokens derived from its identifier and
        declared keywords only (never from docstring prose). Identity matches
        carry full retrieval mass; prose matches are damped by the router and
        planner so descriptive vocabulary cannot outrank another cell's name.
        """
        if self._identity_tokens is None:
            try:
                from .tokenizer import CellTokenizer
            except (ImportError, ValueError):
                from tokenizer import CellTokenizer
            toks = CellTokenizer.tokenize_cell(self.cell_id, self.keywords)
            for p in self.inputs:
                toks.update(CellTokenizer.tokenize_identifier(p))
            for p in self.outputs:
                toks.update(CellTokenizer.tokenize_identifier(p))
            self._identity_tokens = toks
        return self._identity_tokens

    @property
    def is_endable(self) -> bool:
        """
        Determines whether this node can validly terminate an execution pipeline:
        - Explicit node-level endable override if declared.
        - Stage 3 terminal/egress/sink/consumer/evaluator nodes.
        - Display/visualization/cleanup/destructor nodes.
        - Otherwise, intermediate data transformers require downstream closure.
        """
        if self.endable is not None:
            return bool(self.endable)
        if self.stage == 3 or self.node_role in ("sink", "terminal", "evaluator", "consumer"):
            return True
        t_role = self.node_role.lower()
        if t_role in ("display", "visualizer", "cleanup", "destructor"):
            return True
        return False

    @property
    def token_count(self) -> int:
        if self._token_set is None:
            _ = self.token_set
        return self._token_count

    @property
    def primary_input(self) -> PortSignature:
        """
        Identifies the primary data-bearing input port.
        Categorically stage-aware: a Stage 1 ingestion morphism consumes an
        environmental asset, so its textual/path-typed carrier port is primary;
        otherwise the first required non-scalar data port wins, falling back to
        the first declared non-scalar port.
        """
        if self._primary_input is not None:
            return self._primary_input

        if not self.inputs:
            res = PortSignature("input_data", AlgebraicSignature("any", "any"))
            self._primary_input = res
            return res

        registry = TypeRegistry.get_instance()

        def _is_data_carrier(p: PortSignature) -> bool:
            return (
                not registry.is_subtype(p.type_name, "scalar")
                and not registry.is_subtype(p.type_name, "str")
            )

        def _is_asset_carrier(p: PortSignature) -> bool:
            tn = p.type_name.lower()
            return (
                registry.is_subtype(tn, "str")
                or registry.is_subtype(tn, "filepath")
                or registry.is_subtype(tn, "path")
                or registry.is_subtype(tn, "uri")
            )

        stage = getattr(self, "stage", None)
        res = None
        if stage == 1:
            # Stage 1 (Env -> C): the asset carrier is the categorical input.
            asset_ports = [p for p in self.inputs.values() if _is_asset_carrier(p)]
            if asset_ports:
                res = asset_ports[0]
        if res is None:
            required_ports = [p for p in self.inputs.values() if p.required]
            if required_ports:
                required_data = [p for p in required_ports if _is_data_carrier(p)]
                res = required_data[0] if required_data else required_ports[0]
            else:
                data_ports = [p for p in self.inputs.values() if _is_data_carrier(p)]
                res = data_ports[0] if data_ports else next(iter(self.inputs.values()))
        self._primary_input = res
        return res

    @property
    def primary_output(self) -> PortSignature:
        """Identifies the primary data-bearing output port."""
        if self._primary_output is not None:
            return self._primary_output

        if not self.outputs:
            res = PortSignature("output_data", AlgebraicSignature("any", "any"))
            self._primary_output = res
            return res

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
    __slots__ = ("sub_cells", "algorithmic_steps", "_resolved_sub_cells", "internal_topology", "__dict__")

    def __init__(
        self,
        sub_cells: Optional[List[str]] = None,
        algorithmic_steps: Optional[List[str]] = None,
        internal_topology: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        kwargs["cell_type"] = "macro"
        super().__init__(**kwargs)
        self.sub_cells = sub_cells or []
        self.algorithmic_steps = algorithmic_steps or []
        self.internal_topology = internal_topology or {}
        self._resolved_sub_cells: Dict[str, Cell] = {}


class LatticeOrchestrator:
    """
    Mathematical Lattice Topology G = (V, E).
    Maintains nodes V and allows loading/unloading modular knowledge trees.
    An edge (u, v) exists iff u.primary_output unifies with v's accepting input port.
    """
    def __init__(self, trees_directory: str = "trees", active_domain: str = "all", db_path: Optional[str] = None):
        if not os.path.exists(trees_directory) and os.path.exists("new trees"):
            trees_directory = "new trees"
        self.trees_directory = trees_directory
        self.db_path = db_path if db_path is not None else os.path.join(trees_directory, "lattice.db")
        self.active_domain = active_domain
        self.loaded_cells: Dict[str, Cell] = CaseInsensitiveDict()
        self.typestate_vocabularies: Dict[str, Any] = {}
        self._adjacency: Dict[str, List[str]] = {}
        self._reverse_adjacency: Dict[str, List[str]] = {}
        self._token_index: Dict[str, List[Cell]] = {}
        self._bridge_cells: List[Cell] = []
        self.dynamic_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._lock = threading.RLock()

        if os.path.exists(self.db_path):
            self.load_from_database(self.db_path)
        else:
            self.load_all_json_trees()
        self.build_topology()

    @property
    def cells(self) -> List[Cell]:
        return list(self.loaded_cells.values())

    @property
    def _cells_by_input(self) -> Dict[Tuple[str, str], List[Cell]]:
        res: Dict[Tuple[str, str], List[Cell]] = {}
        for c in self.loaded_cells.values():
            for p in c.inputs.values():
                res.setdefault((p.type_name, p.state), []).append(c)
        return res

    @property
    def _cells_by_output(self) -> Dict[Tuple[str, str], List[Cell]]:
        res: Dict[Tuple[str, str], List[Cell]] = {}
        for c in self.loaded_cells.values():
            for p in c.outputs.values():
                res.setdefault((p.type_name, p.state), []).append(c)
        return res

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
                if "types" in data and isinstance(data["types"], dict):
                    reg = TypeRegistry.get_instance()
                    for t_name, t_meta in data["types"].items():
                        parent = t_meta.get("parent") if isinstance(t_meta, dict) else str(t_meta)
                        if parent:
                            reg.register_type(t_name, parent)
                if "typestates" in data and data["typestates"]:
                    self.typestate_vocabularies[domain] = data["typestates"]
                    ts_info = data["typestates"]
                    registry = TypeRegistry.get_instance()
                    states_list = ts_info.get("states", []) if isinstance(ts_info, dict) else []
                    for s in states_list:
                        if isinstance(s, dict):
                            s_name = s.get("name")
                            s_parent = s.get("parent_state")
                            if s_name:
                                registry.register_state(s_name, s_parent)
                        elif isinstance(s, str):
                            registry.register_state(s)

            for c_dict in raw_cells:
                is_macro = (
                    str(c_dict.get("node_type", "")).lower() in ("macro", "higher_order")
                    or str(c_dict.get("node_role", "")).lower() in ("macro", "higher_order")
                    or str(c_dict.get("node_type", "")).lower().startswith("macro_")
                    or str(c_dict.get("node_role", "")).lower().startswith("macro_")
                )
                cell_cls = MacroCell if is_macro else MicroCell
                cell = cell_cls(
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
                    is_public=bool(c_dict.get("is_public", True)),
                    mutation_type=c_dict.get("mutation_type", "pure"),
                    is_context_manager=bool(c_dict.get("is_context_manager", False)),
                    raises=c_dict.get("raises", []),
                    type_vars=c_dict.get("type_vars", []),
                    preconditions=c_dict.get("preconditions", []),
                    postconditions=c_dict.get("postconditions", []),
                    effects=c_dict.get("effects", []),
                    edges=c_dict.get("edges", []),
                )
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
        target_dir = self.trees_directory
        if not os.path.exists(target_dir):
            if os.path.exists("new trees"):
                target_dir = "new trees"
            else:
                return
        all_fnames = [f for f in os.listdir(target_dir) if f.endswith(".json")]
        normalized_fnames = [f for f in all_fnames if f.endswith("_normalized.json")]
        target_fnames = normalized_fnames if normalized_fnames else all_fnames
        for fname in sorted(target_fnames):
            self.load_tree_file(os.path.join(target_dir, fname))

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
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='types'")
                if cursor.fetchone():
                    cursor.execute("SELECT type_name, parent_type FROM types")
                    reg = TypeRegistry.get_instance()
                    for t_name, p_type in cursor.fetchall():
                        if t_name and p_type:
                            reg.register_type(t_name, p_type)
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
                    slots_sel = "slots" if "slots" in col_names else "'' AS slots"

                    cursor.execute(f"""
                        SELECT cell_id, {dom_sel}, {type_sel}, {role_sel}, stage,
                               keywords, input_type, input_state, output_type, output_state,
                               code, {deps_sel}, {cfg_sel}, {slots_sel}, {ver_sel}, {doc_sel}, {prio_sel}
                        FROM nodes
                    """)
                    for row in cursor.fetchall():
                        (cell_id, domain_name, node_type, node_role, stage,
                         keywords_json, in_type, in_state, out_type, out_state,
                         code, deps_json, config_json, slots_json, verified, doc_str, source_priority) = row

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

                        slots_val = cfg.get("slots", {})
                        if not slots_val and slots_json:
                            try:
                                slots_val = json.loads(slots_json)
                            except Exception:
                                slots_val = {}

                        in_sig = AlgebraicSignature(type_name=in_type or "any", state=in_state or "any")
                        out_sig = AlgebraicSignature(type_name=out_type or "None", state=out_state or "any")

                        inputs: Dict[str, PortSignature] = {}
                        outputs: Dict[str, PortSignature] = {}

                        if isinstance(cfg, dict) and ("inputs" in cfg or "outputs" in cfg):
                            for p_name, p_val in cfg.get("inputs", {}).items():
                                if isinstance(p_val, dict):
                                    abs_t = _clean_abs_carrier(p_val.get("abstract_type"))
                                    acc_s = frozenset(str(s).strip().lower() for s in (p_val.get("accepted_states") or []) if str(s).strip())
                                    p_s = str(p_val.get("parent_state") or "").strip().lower() or None
                                    if p_s and p_val.get("state"):
                                        TypeRegistry.get_instance().register_state(p_val.get("state"), p_s)
                                    inputs[p_name] = PortSignature(
                                        name=p_name,
                                        signature=AlgebraicSignature(
                                            type_name=str(p_val.get("type_name", in_type or "any")),
                                            state=str(p_val.get("state", in_state or "any")),
                                            qualifiers=_normalize_qualifiers(p_val.get("qualifiers", [])),
                                            abstract_type=abs_t,
                                            accepted_states=acc_s,
                                            parent_state=p_s,
                                        ),
                                        required=p_val.get("required", True),
                                        default_value=p_val.get("default_value"),
                                        domain=p_val.get("domain", ""),
                                        abstract_type=abs_t,
                                        enum_values=p_val.get("enum_values"),
                                        param_kind=p_val.get("param_kind", "standard"),
                                        value_constraints=p_val.get("value_constraints"),
                                        shape_contract=p_val.get("shape_contract"),
                                        accepted_states=acc_s,
                                        parent_state=p_s,
                                        port_role=p_val.get("port_role"),
                                    )
                            for p_name, p_val in cfg.get("outputs", {}).items():
                                if isinstance(p_val, dict):
                                    abs_t = _clean_abs_carrier(p_val.get("abstract_type"))
                                    acc_s = frozenset(str(s).strip().lower() for s in (p_val.get("accepted_states") or []) if str(s).strip())
                                    p_s = str(p_val.get("parent_state") or "").strip().lower() or None
                                    if p_s and p_val.get("state"):
                                        TypeRegistry.get_instance().register_state(p_val.get("state"), p_s)
                                    outputs[p_name] = PortSignature(
                                        name=p_name,
                                        signature=AlgebraicSignature(
                                            type_name=str(p_val.get("type_name", out_type or "any")),
                                            state=str(p_val.get("state", out_state or "any")),
                                            qualifiers=_normalize_qualifiers(p_val.get("qualifiers", [])),
                                            abstract_type=abs_t,
                                            accepted_states=acc_s,
                                            parent_state=p_s,
                                        ),
                                        required=p_val.get("required", True),
                                        default_value=p_val.get("default_value"),
                                        domain=p_val.get("domain", ""),
                                        abstract_type=abs_t,
                                        enum_values=p_val.get("enum_values"),
                                        param_kind=p_val.get("param_kind", "standard"),
                                        value_constraints=p_val.get("value_constraints"),
                                        shape_contract=p_val.get("shape_contract"),
                                        accepted_states=acc_s,
                                        parent_state=p_s,
                                        port_role=p_val.get("port_role"),
                                    )

                        if not inputs and (not isinstance(cfg, dict) or "inputs" not in cfg):
                            inputs = {"input_data": PortSignature("input_data", in_sig)}
                        if not outputs and (not isinstance(cfg, dict) or "outputs" not in cfg):
                            outputs = {"output_data": PortSignature("output_data", out_sig)}

                        is_macro = (
                            str(node_type).lower() in ("macro", "higher_order")
                            or str(node_role).lower() in ("macro", "higher_order")
                            or str(node_type).lower().startswith("macro_")
                            or str(node_role).lower().startswith("macro_")
                        )
                        cls = MacroCell if is_macro else MicroCell
                        cell = cls(
                            cell_id=cell_id,
                            stage=stage or 2,
                            keywords=keywords,
                            inputs=inputs,
                            outputs=outputs,
                            domain_name=domain_name or "generic",
                            node_type=node_type or "function",
                            node_role=node_role or "function",
                            dependencies=deps,
                            code_template=code or "",
                            metadata_tags=cfg.get("metadata_tags", {}),
                            configuration_schema=cfg,
                            slots=slots_val,
                            verified=bool(verified),
                            docstring=doc_str or "",
                            source_priority=int(source_priority) if source_priority is not None else 100,
                            source_provenance=f"sqlite:{Path(self.db_path).name}",
                            topology_type=cfg.get("topology_type", "sequential"),
                            feedback_state_type=cfg.get("feedback_state_type"),
                            bound_slots=cfg.get("bound_slots", {}),
                            is_public=bool(cfg.get("is_public", True)),
                            mutation_type=cfg.get("mutation_type", "pure"),
                            is_context_manager=cfg.get("is_context_manager", False),
                            raises=cfg.get("raises", []),
                            type_vars=cfg.get("type_vars", []),
                            preconditions=cfg.get("preconditions", []),
                            postconditions=cfg.get("postconditions", []),
                            effects=cfg.get("effects", []),
                            edges=cfg.get("edges", []),
                            endable=cfg.get("endable")
                        )
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
                                            qualifiers=_normalize_qualifiers(p_val.get("qualifiers", []))
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
                                            qualifiers=_normalize_qualifiers(p_val.get("qualifiers", []))
                                        )
                                    )

                        if not inputs:
                            inputs = {"input_data": PortSignature("input_data", in_sig)}
                        if not outputs:
                            outputs = {"output_data": PortSignature("output_data", out_sig)}

                        keywords = CellTokenizer.tokenize_identifier(cell_id)

                        cell = MicroCell(
                            cell_id=cell_id,
                            stage=stage or 2,
                            keywords=keywords,
                            inputs=inputs,
                            outputs=outputs,
                            domain_name="data_processing",
                            dependencies=deps,
                            code_template=code or "",
                            verified=True,
                            endable=cfg.get("endable")
                        )
                        if cell.is_public_morphism:
                            self.loaded_cells[cell.cell_id] = cell

                conn.close()
                logger.info(f"[LATTICE] Loaded {len(self.loaded_cells)} nodes from database.")
            except Exception as e:
                logger.error(f"[LATTICE] Failed loading database: {e}")

    def build_topology(self):
        """
        Builds the lattice directed edges based strictly on declared edges and monadic type compatibility:
          (u, v) in E <=> v.cell_id in u.edges or u.primary_output.unifies_with(v.primary_input)
        """
        with self._lock:
            self._adjacency.clear()
            self._reverse_adjacency.clear()
            self._token_index.clear()
            self._bridge_cells.clear()

            all_cells = list(self.loaded_cells.values())
            for cell in all_cells:
                _ = cell.token_set  # Warm up cached token set
                for tok in cell.token_set:
                    self._token_index.setdefault(tok, []).append(cell)
                if getattr(cell, "node_role", "") == "bridge" or getattr(cell, "node_type", "") == "tunnel":
                    self._bridge_cells.append(cell)
                self._adjacency[cell.cell_id] = []
                self._reverse_adjacency[cell.cell_id] = []

            # Populate graph edges
            for u in all_cells:
                # 1. Declared edges from cell schema
                for edge in getattr(u, "edges", []):
                    tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                    if tgt_id:
                        tgt_cell = self.loaded_cells.get(tgt_id)
                        if tgt_cell and tgt_cell.cell_id not in self._adjacency[u.cell_id]:
                            self._adjacency[u.cell_id].append(tgt_cell.cell_id)
                            self._reverse_adjacency[tgt_cell.cell_id].append(u.cell_id)

                # 2. Monadic type compatibility (guarded against undefined/none/any spurious edges)
                out_p = u.primary_output
                if out_p and getattr(out_p, "type_name", None):
                    out_tn = str(out_p.type_name).strip().lower()
                    if out_tn not in ("none", "null", "undefined", "any", "*", ""):
                        out_sig = out_p.signature
                        is_out_bot = (hasattr(out_sig, 'is_bottom') and out_sig.is_bottom()) or out_tn in ('none', 'null', 'undefined', 'bottom', '')
                        if not (hasattr(out_sig, 'is_top') and out_sig.is_top()) and not is_out_bot:
                            for v in all_cells:
                                if u.cell_id == v.cell_id or v.cell_id in self._adjacency[u.cell_id]:
                                    continue
                                # Stage 1 source/reader nodes never accept upstream incoming dataflow
                                if getattr(v, "stage", None) == 1 or not getattr(v, "inputs", None):
                                    continue
                                in_p = v.primary_input
                                if in_p and getattr(in_p, "type_name", None):
                                    in_tn = str(in_p.type_name).strip().lower()
                                    if in_tn not in ("none", "null", "undefined", "any", "*", ""):
                                        in_sig = in_p.signature
                                        is_in_bot = (hasattr(in_sig, 'is_bottom') and in_sig.is_bottom()) or in_tn in ('none', 'null', 'undefined', 'bottom', '')
                                        if not (hasattr(in_sig, 'is_top') and in_sig.is_top()) and not is_in_bot:
                                            if out_sig.unifies_with(in_sig):
                                                self._adjacency[u.cell_id].append(v.cell_id)
                                                self._reverse_adjacency[v.cell_id].append(u.cell_id)

    @property
    def token_index(self) -> Dict[str, List[Cell]]:
        return self._token_index

    @property
    def bridge_cells(self) -> List[Cell]:
        return self._bridge_cells

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
        for c in self.loaded_cells.values():
            if c.can_accept(sig):
                results.append(c)
        return results

    def register_dynamic_edge(
        self,
        src_cell_id: str,
        dst_cell_id: str,
        affinity_score: float = 0.8,
        provenance: str = "runtime_verified"
    ) -> bool:
        """
        Dynamically promotes a verified cross-tree or within-tree bridge morphism edge at runtime.
        Updates in-memory adjacency, adds to source cell edges, and tracks in dynamic_edges.
        """
        with self._lock:
            src_cell = self.loaded_cells.get(src_cell_id)
            dst_cell = self.loaded_cells.get(dst_cell_id)
            if not src_cell or not dst_cell:
                return False

            edge_data = {
                "target_cell_id": dst_cell.cell_id,
                "affinity_score": float(affinity_score),
                "score_provenance": provenance,
                "is_cross_tree": bool(src_cell.domain_name != dst_cell.domain_name)
            }
            self.dynamic_edges[(src_cell.cell_id, dst_cell.cell_id)] = edge_data

            # Add to cell edges if not already present
            existing_targets = {
                e.get("target_cell_id") if isinstance(e, dict) else getattr(e, "target_cell_id", None)
                for e in getattr(src_cell, "edges", [])
            }
            if dst_cell.cell_id not in existing_targets:
                if not hasattr(src_cell, "edges") or src_cell.edges is None:
                    src_cell.edges = []
                src_cell.edges.append(edge_data)

            # Add to directed adjacency
            if src_cell.cell_id not in self._adjacency:
                self._adjacency[src_cell.cell_id] = []
            if dst_cell.cell_id not in self._adjacency[src_cell.cell_id]:
                self._adjacency[src_cell.cell_id].append(dst_cell.cell_id)

            if dst_cell.cell_id not in self._reverse_adjacency:
                self._reverse_adjacency[dst_cell.cell_id] = []
            if src_cell.cell_id not in self._reverse_adjacency[dst_cell.cell_id]:
                self._reverse_adjacency[dst_cell.cell_id].append(src_cell.cell_id)

            return True

    def get_promoted_bridges(self) -> List[Dict[str, Any]]:
        """Returns all dynamically promoted bridge edges."""
        with self._lock:
            return [
                {"source": s, "target": t, **data}
                for (s, t), data in self.dynamic_edges.items()
            ]
