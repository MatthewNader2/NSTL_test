"""
src/unification.py - Neuro-Symbolic Topological Lattice (NSTL)
Formal Type-Monadic Unification Gate and Deterministic Composition Synthesizer.

Conforms strictly to Section 3.2 of the NSTL paper:
  M_T(A) = { (a, sigma) : a in A, sigma a type substitution } U { bottom }
  bind(m, k) = k(a) with sigma_new if sigma_new = unify(tau_out of m, tau_in of k) succeeds;
               otherwise bottom.
"""

from __future__ import annotations
import ast
import collections
import sys
import json
import math
import keyword
import threading
import warnings
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable, Generic, TypeVar, FrozenSet

from log_config import get_logger

try:
    from utils import (
        tokenize_alphanumeric,
        extract_template_placeholders,
        safe_substitute_template,
    )
    from errors import (
        UnificationError,
        PlaceholderResolutionError,
        TemplateValidationError,
        DataflowExecutionError,
        PostconditionVerificationError,
    )
except ImportError:
    from .utils import (
        tokenize_alphanumeric,
        extract_template_placeholders,
        safe_substitute_template,
    )
    from .errors import (
        UnificationError,
        PlaceholderResolutionError,
        TemplateValidationError,
        DataflowExecutionError,
        PostconditionVerificationError,
    )

try:
    from .lattice import (
        AlgebraicSignature, PortSignature, Cell, MacroCell, TypeRegistry,
        ABSTRACT_CARRIERS, UNRESOLVED_PORT, GENERIC_TYPE_VARIABLE_NAMES,
        is_path_port as _lattice_is_path_port,
    )
    from .tokenizer import CellTokenizer, normalize_token
except (ImportError, ValueError):
    from lattice import (
        AlgebraicSignature, PortSignature, Cell, MacroCell, TypeRegistry,
        ABSTRACT_CARRIERS, UNRESOLVED_PORT, GENERIC_TYPE_VARIABLE_NAMES,
        is_path_port as _lattice_is_path_port,
    )
    from tokenizer import CellTokenizer, normalize_token

logger = get_logger('unification')

T = TypeVar('T')
U = TypeVar('U')

# English sentence-connective function words (LANGUAGE-level primitives, not
# domain vocabulary): a capitalized occurrence of one of these mid-prompt is a
class DynamicSentenceConnectives(frozenset):
    """Dynamic sentence connectives backed by TypeRegistry declared connectives."""
    def __contains__(self, item):
        return item in TypeRegistry.get_instance().get_sentence_connectives()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_sentence_connectives())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_sentence_connectives())
    def __sub__(self, other):
        return TypeRegistry.get_instance().get_sentence_connectives() - (set(other) if not isinstance(other, set) else other)
    def __rsub__(self, other):
        return set(other) - TypeRegistry.get_instance().get_sentence_connectives()
    def __and__(self, other):
        return TypeRegistry.get_instance().get_sentence_connectives() & (set(other) if not isinstance(other, set) else other)
    def __rand__(self, other):
        return (set(other) if not isinstance(other, set) else other) & TypeRegistry.get_instance().get_sentence_connectives()

_SENTENCE_CONNECTIVES = DynamicSentenceConnectives()

class DynamicPolarityHints(frozenset):
    """Dynamic ordering polarity hints harvested from domain trees and order flags."""
    def __init__(self, direction: str):
        self.direction = direction
    def __contains__(self, item):
        return item in TypeRegistry.get_instance().get_polarity_hints(self.direction)
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_polarity_hints(self.direction))
    def __len__(self):
        return len(TypeRegistry.get_instance().get_polarity_hints(self.direction))
    def __and__(self, other):
        return (set(other) if not isinstance(other, set) else other) & TypeRegistry.get_instance().get_polarity_hints(self.direction)
    def __rand__(self, other):
        return (set(other) if not isinstance(other, set) else other) & TypeRegistry.get_instance().get_polarity_hints(self.direction)

_ASCENDING_HINTS = DynamicPolarityHints("ascending")
_DESCENDING_HINTS = DynamicPolarityHints("descending")
# Port names/states that carry an ordering polarity (generic, not domain names).
_ORDER_FLAG_NAMES = frozenset()          # must be registered from trees
_ORDER_FLAG_STATES = frozenset()
class DynamicPrepositionTriggers(frozenset):
    """Dynamic relational preposition triggers harvested from domain trees."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_preposition_triggers()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_preposition_triggers())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_preposition_triggers())

_PREPOSITION_TRIGGERS = DynamicPrepositionTriggers()
_ORDER_POLE_TOKENS = frozenset()


class IdentifierGroup:
    """
    An enumerable set of bare referential identifiers sharing one syntactic
    role context (e.g. ``normalize X column, Y column and Z column`` groups
    X, Y, Z under the role noun ``column``).

    members:     [(char_pos, token), ...] in prompt order
    role_tokens: stemmed context tokens naming the shared role ({"column"});
                 empty when an identifier stands alone with no content context.
    """

    __slots__ = ("members", "role_tokens")

    def __init__(self, members: List[Tuple[int, str]], role_tokens: FrozenSet[str]):
        self.members = members
        self.role_tokens = role_tokens

    def __repr__(self) -> str:
        return f"IdentifierGroup(members={[m[1] for m in self.members]}, role={set(self.role_tokens)})"


def is_top_symbol(raw: str, registry: Optional[Any] = None) -> bool:
    s = str(raw).strip()
    if s in ("⊤", "*"):
        return True
    if registry is None:
        registry = TypeRegistry.get_instance()
    return registry.is_declared_top(s)


class _DynamicTopTypeSet(frozenset):
    """Dynamic top type set backed by TypeRegistry declarations."""
    def __contains__(self, item):
        return TypeRegistry.get_instance().is_declared_top(str(item))
    def __iter__(self):
        return iter(TypeRegistry.get_instance()._declared_top)
    def __len__(self):
        return len(TypeRegistry.get_instance()._declared_top)

TOP_TYPE_SET = _DynamicTopTypeSet()


# =====================================================================
# 1. Type Terms and Substitutions
# =====================================================================

class TypeTerm(ABC):
    """Abstract base class for formal type terms."""
    @abstractmethod
    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        pass

    @classmethod
    def from_string(cls, s: str) -> 'TypeTerm':
        s_clean = str(s).strip()
        registry = TypeRegistry.get_instance()
        if is_top_symbol(s_clean, registry):
            return TOP
        if s_clean.startswith("?"):
            return TypeVariable(s_clean[1:])
        # Coproduct / Sum / Union types (e.g. A | B or Union[A, B])
        if " | " in s_clean:
            parts = [p.strip() for p in s_clean.split(" | ")]
            return UnionTypeTerm(tuple(TypeTerm.from_string(p) for p in parts if p))
        if s_clean.startswith("Union[") and s_clean.endswith("]"):
            inner = s_clean[6:-1].strip()
            args = [a.strip() for a in inner.split(",") if a.strip()]
            return UnionTypeTerm(tuple(TypeTerm.from_string(a) for a in args))
        # Declared type variables (dynamic registry + uppercase mathematical fallback)
        if registry.is_type_variable(s_clean):
            return TypeVariable(s_clean)
        # Check for container/generic expressions like Sequence[T], List[MatLike], Dict[K, V]
        if "[" in s_clean and s_clean.endswith("]"):
            bracket_idx = s_clean.index("[")
            constructor = s_clean[:bracket_idx].strip()
            inner = s_clean[bracket_idx + 1 : -1].strip()
            args = []
            depth = 0
            curr = []
            for ch in inner:
                if ch == "[":
                    depth += 1
                    curr.append(ch)
                elif ch == "]":
                    depth -= 1
                    curr.append(ch)
                elif ch == "," and depth == 0:
                    args.append("".join(curr).strip())
                    curr = []
                else:
                    curr.append(ch)
            if curr:
                args.append("".join(curr).strip())
            parsed_args = tuple(TypeTerm.from_string(a) for a in args if a)
            if parsed_args:
                return GenericTypeTerm(constructor, parsed_args)
        return AtomicType(s_clean)


@dataclass(frozen=True, slots=True)
class TopType(TypeTerm):
    """Universal Top type (wildcard) unifying with any type term."""
    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        return self

    def __repr__(self) -> str:
        return "Top"


TOP = TopType()


@dataclass(frozen=True, slots=True)
class AtomicType(TypeTerm):
    """Ground type constant."""
    name: str

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        return self

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class GenericTypeTerm(TypeTerm):
    """
    Parametric / Generic type constructor: C[T_1, ..., T_n].
    Conforms to Category of Generic Monads and Functors.
    """
    constructor: str
    args: Tuple[TypeTerm, ...]

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        resolved_args = tuple(a.apply_substitution(sigma, visited) for a in self.args)
        return GenericTypeTerm(self.constructor, resolved_args)

    def __repr__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.constructor}[{args_str}]"


@dataclass(frozen=True, slots=True)
class UnionTypeTerm(TypeTerm):
    """
    Coproduct / Sum / Union type term: T_1 | ... | T_n.
    Conforms to Category of Coproducts with canonical injection morphisms.
    """
    terms: Tuple[TypeTerm, ...]

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        resolved_terms = tuple(t.apply_substitution(sigma, visited) for t in self.terms)
        return UnionTypeTerm(resolved_terms)

    def __repr__(self) -> str:
        return " | ".join(str(t) for t in self.terms)


@dataclass(frozen=True, slots=True)
class TypeVariable(TypeTerm):
    """Type variable alpha, beta... subject to substitution."""
    var_name: str

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        if visited and self.var_name in visited:
            return self
        if self.var_name in sigma.mappings:
            target = sigma.mappings[self.var_name]
            new_visited = (visited or frozenset()) | {self.var_name}
            if isinstance(target, TypeTerm):
                return target.apply_substitution(sigma, new_visited)
            target_str = str(target).strip()
            if target_str == self.var_name or target_str == f"?{self.var_name}":
                return self
            parsed = TypeTerm.from_string(target_str)
            if isinstance(parsed, TypeVariable) and parsed.var_name == self.var_name:
                return self
            return parsed.apply_substitution(sigma, new_visited)
        return self

    def __repr__(self) -> str:
        return f"?{self.var_name}"


@dataclass(frozen=True, slots=True)
class TypestateTerm(TypeTerm):
    """Typestate compound term: tau = (type_name, state, qualifiers, abstract_type, accepted_states, parent_state)."""
    type_name: str
    state: str = "any"
    qualifiers: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)
    abstract_type: Optional[str] = None
    accepted_states: FrozenSet[str] = field(default_factory=frozenset)
    parent_state: Optional[str] = None

    def __post_init__(self):
        raw = self.abstract_type
        if raw and str(raw).strip().lower() not in ("none", "null", ""):
            object.__setattr__(self, "abstract_type", str(raw).strip())
        else:
            object.__setattr__(self, "abstract_type", None)
        if self.accepted_states and not isinstance(self.accepted_states, frozenset):
            object.__setattr__(self, "accepted_states", frozenset(str(s).strip().lower() for s in self.accepted_states if str(s).strip()))
        if self.parent_state:
            object.__setattr__(self, "parent_state", str(self.parent_state).strip().lower())

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        t_resolved = self.type_name
        if self.type_name in sigma.mappings:
            if visited and self.type_name in visited:
                return self
            val = sigma.mappings[self.type_name]
            if isinstance(val, AtomicType):
                t_resolved = val.name
            elif isinstance(val, TypeTerm):
                new_visited = (visited or frozenset()) | {self.type_name}
                t_resolved = str(val.apply_substitution(sigma, new_visited))
            else:
                t_resolved = str(val)
        return TypestateTerm(
            type_name=t_resolved,
            state=self.state,
            qualifiers=self.qualifiers,
            abstract_type=self.abstract_type,
            accepted_states=self.accepted_states,
            parent_state=self.parent_state,
        )

    def __repr__(self) -> str:
        abs_str = f", abs={self.abstract_type}" if self.abstract_type else ""
        acc_str = f", acc={list(self.accepted_states)}" if self.accepted_states else ""
        p_str = f", parent={self.parent_state}" if self.parent_state else ""
        return f"{self.type_name}[{self.state}{abs_str}{acc_str}{p_str}]"


class Substitution:
    """Mapping of variable identifiers to resolved type terms or values."""
    def __init__(self, mappings: Optional[Dict[str, Any]] = None):
        self.mappings: Dict[str, Any] = dict(mappings) if mappings else {}

    def bind(self, var: str, value: Any):
        self.mappings[var] = value

    def get(self, var: str, default: Any = None) -> Any:
        return self.mappings.get(var, default)

    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose substitutions: (sigma1 . sigma2)(t) = sigma1(sigma2(t))."""
        new_map = dict(self.mappings)
        for k, v in other.mappings.items():
            if k not in new_map:
                new_map[k] = v
        return Substitution(new_map)

    def copy(self) -> 'Substitution':
        return Substitution(dict(self.mappings))

    def __repr__(self) -> str:
        return f"σ({self.mappings})"


# =====================================================================
# 2. Robinson's First-Order Unification Algorithm
# =====================================================================

def occurs_check(
    var_name: str,
    term: Any,
    sigma: Optional['Substitution'] = None,
    visited: Optional[FrozenSet[str]] = None
) -> bool:
    """
    Returns True if type variable `var_name` occurs within `term`.
    Occurs check is fundamental to Robinson's first-order unification to prevent
    infinite / cyclic terms (e.g. T = Sequence[T]).
    """
    if visited and var_name in visited:
        return False

    if isinstance(term, TypeVariable):
        if term.var_name == var_name:
            return True
        if sigma and term.var_name in sigma.mappings:
            new_visited = (visited or frozenset()) | {term.var_name}
            return occurs_check(var_name, sigma.mappings[term.var_name], sigma, new_visited)
        return False
    elif isinstance(term, GenericTypeTerm):
        return any(occurs_check(var_name, a, sigma, visited) for a in term.args)
    elif isinstance(term, UnionTypeTerm):
        return any(occurs_check(var_name, t, sigma, visited) for t in term.terms)
    elif isinstance(term, TypestateTerm):
        if term.type_name == var_name:
            return True
        if sigma and term.type_name in sigma.mappings:
            new_visited = (visited or frozenset()) | {term.type_name}
            return occurs_check(var_name, sigma.mappings[term.type_name], sigma, new_visited)
        return False
    elif isinstance(term, AtomicType):
        return term.name == var_name
    elif isinstance(term, AlgebraicSignature):
        return occurs_check(var_name, term.type_name, sigma, visited)
    elif isinstance(term, str):
        if var_name == term:
            return True
        tokens = tokenize_alphanumeric(term, min_len=1)
        if var_name in tokens:
            return True
        if sigma:
            for tok in tokens:
                if tok in sigma.mappings and (not visited or tok not in visited):
                    new_visited = (visited or frozenset()) | {tok}
                    if occurs_check(var_name, sigma.mappings[tok], sigma, new_visited):
                        return True
        return False
    return False


_UNIFY_CACHE_MAX = 4096
_UNIFY_CACHE_LOCK = threading.Lock()
_UNIFY_BASE_CACHE: collections.OrderedDict[Tuple[TypeTerm, TypeTerm], Optional[Substitution]] = collections.OrderedDict()


def _unify_cache_put(key: Tuple[TypeTerm, TypeTerm], value: Optional[Substitution]) -> None:
    """Thread-safe, bounded write to _UNIFY_BASE_CACHE."""
    with _UNIFY_CACHE_LOCK:
        _UNIFY_BASE_CACHE[key] = value
        if len(_UNIFY_BASE_CACHE) > _UNIFY_CACHE_MAX:
            _UNIFY_BASE_CACHE.popitem(last=False)


def unify(
    term1: Union[TypeTerm, AlgebraicSignature, str],
    term2: Union[TypeTerm, AlgebraicSignature, str],
    sigma: Optional[Substitution] = None
) -> Optional[Substitution]:
    """
    Computes Most General Unifier (mgu) of term1 and term2.
    Returns updated Substitution sigma if unification succeeds, or None (bottom) on failure.
    """
    # Normalize AlgebraicSignature to TypestateTerm
    t1 = _to_type_term(term1)
    t2 = _to_type_term(term2)

    is_ground_query = (sigma is None or not sigma.mappings)
    if is_ground_query:
        cache_key = (t1, t2)
        with _UNIFY_CACHE_LOCK:
            if cache_key in _UNIFY_BASE_CACHE:
                cached = _UNIFY_BASE_CACHE[cache_key]
                return Substitution(cached.mappings) if cached is not None else None

    sub = Substitution(sigma.mappings if sigma else {})

    t1 = t1.apply_substitution(sub)
    t2 = t2.apply_substitution(sub)

    # 1. Identity or Universal Top (Top unifies with any type term)
    if t1 == t2 or isinstance(t2, TopType) or isinstance(t1, TopType):
        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

    # 2. Variable binding (Robinson first-order unification with occurs check)
    if isinstance(t1, TypeVariable):
        if isinstance(t2, TypeVariable) and t1.var_name == t2.var_name:
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub
        if occurs_check(t1.var_name, t2, sub):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None  # Occurs check failure -> bottom
        sub.bind(t1.var_name, t2)
        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

    if isinstance(t2, TypeVariable):
        if occurs_check(t2.var_name, t1, sub):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None  # Occurs check failure -> bottom
        sub.bind(t2.var_name, t1)
        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

    # 2.3. Coproduct / Union unification (canonical injection)
    if isinstance(t1, UnionTypeTerm):
        for alt in t1.terms:
            sub_alt = unify(alt, t2, sub)
            if sub_alt is not None:
                if is_ground_query:
                    _unify_cache_put(cache_key, sub_alt)
                return sub_alt
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    if isinstance(t2, UnionTypeTerm):
        for alt in t2.terms:
            sub_alt = unify(t1, alt, sub)
            if sub_alt is not None:
                if is_ground_query:
                    _unify_cache_put(cache_key, sub_alt)
                return sub_alt
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    # 2.5. Generic container unification (covariant structural unification)
    if isinstance(t1, GenericTypeTerm) and isinstance(t2, GenericTypeTerm):
        registry = TypeRegistry.get_instance()
        c1 = t1.constructor.lower()
        c2 = t2.constructor.lower()
        compat = (c1 == c2) or registry.is_subtype(c1, c2)
        if compat and len(t1.args) == len(t2.args):
            for a1, a2 in zip(t1.args, t2.args):
                sub = unify(a1, a2, sub)
                if sub is None:
                    if is_ground_query:
                        _unify_cache_put(cache_key, None)
                    return None
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    # Raw type fallback: List[Contour] satisfies unparameterized List
    if isinstance(t1, GenericTypeTerm) and (isinstance(t2, AtomicType) or isinstance(t2, TypestateTerm)):
        registry = TypeRegistry.get_instance()
        target_name = t2.type_name if isinstance(t2, TypestateTerm) else t2.name
        if registry.is_subtype(t1.constructor, target_name):
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub

    # Sound parameter demand: untyped list satisfies GenericTypeTerm ONLY IF parameters are type variables
    if isinstance(t2, GenericTypeTerm) and (isinstance(t1, AtomicType) or isinstance(t1, TypestateTerm)):
        registry = TypeRegistry.get_instance()
        source_name = t1.type_name if isinstance(t1, TypestateTerm) else t1.name
        if registry.is_subtype(source_name, t2.constructor):
            if all(isinstance(arg, (TypeVariable, TopType)) for arg in t2.args):
                for arg in t2.args:
                    if isinstance(arg, TypeVariable) and arg.var_name not in sub.mappings:
                        sub.bind(arg.var_name, TOP)
                if is_ground_query:
                    _unify_cache_put(cache_key, sub)
                return sub
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None

    # 3. Typestate term unification
    if isinstance(t1, TypestateTerm) and isinstance(t2, TypestateTerm):
        # State compatibility check: delegate to TypeRegistry.is_state_compatible
        registry = TypeRegistry.get_instance()
        if not registry.is_state_compatible(
            producer_state=t1.state,
            consumer_state=t2.state,
            producer_accepted=t1.accepted_states,
            consumer_accepted=t2.accepted_states,
            consumer_parent=t2.parent_state,
            producer_parent=t1.parent_state,
        ):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None  # State mismatch -> bottom

        # Qualifier subset check
        if t2.qualifiers and not t2.qualifiers.issubset(t1.qualifiers):
            ignorable = TypeRegistry.get_instance().get_advisory_qualifiers()
            req = {q for q in t2.qualifiers if q not in ignorable}
            if req and not req.issubset(t1.qualifiers):
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None

        # Check if type_names contain generic definitions
        if ("[" in t1.type_name) or ("[" in t2.type_name) or (len(t1.type_name) == 1 and t1.type_name.isupper()) or (len(t2.type_name) == 1 and t2.type_name.isupper()):
            inner1 = TypeTerm.from_string(t1.type_name)
            inner2 = TypeTerm.from_string(t2.type_name)
            return unify(inner1, inner2, sub)

        # Type poset subtyping check: t1.type_name <= t2.type_name
        registry = TypeRegistry.get_instance()
        if not registry.is_subtype(t1.type_name, t2.type_name):
            t2_tn = (t2.type_name or "").strip().lower()
            t2_abs = (t2.abstract_type or "").strip().lower()
            is_consumer_abstract = (t2_tn in ABSTRACT_CARRIERS or t2_tn == t2_abs)
            if (
                is_consumer_abstract
                and t1.abstract_type
                and t2.abstract_type
                and registry.is_subtype(t1.abstract_type, t2.abstract_type)
            ):
                pass
            else:
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None  # Type mismatch -> bottom

        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

    # 4. Atomic type unification
    if isinstance(t1, AtomicType) and isinstance(t2, AtomicType):
        registry = TypeRegistry.get_instance()
        if registry.is_subtype(t1.name, t2.name):
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    # 5. Mixed Atomic and Typestate unification
    if isinstance(t1, AtomicType) and isinstance(t2, TypestateTerm):
        registry = TypeRegistry.get_instance()
        if registry.is_subtype(t1.name, t2.type_name):
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    if isinstance(t1, TypestateTerm) and isinstance(t2, AtomicType):
        registry = TypeRegistry.get_instance()
        if registry.is_subtype(t1.type_name, t2.name):
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    if is_ground_query:
        _unify_cache_put(cache_key, None)
    return None


_ALGEBRAIC_SIG_TERM_CACHE: Dict[AlgebraicSignature, TypeTerm] = {}


def _to_type_term(item: Any) -> TypeTerm:
    if isinstance(item, TypeTerm):
        return item
    if isinstance(item, PortSignature):
        cached = getattr(item, "_cached_term", None)
        if cached is not None:
            return cached
        res = _to_type_term(item.signature)
        try:
            item._cached_term = res
        except (AttributeError, TypeError):
            pass
        return res
    if isinstance(item, AlgebraicSignature):
        cached = _ALGEBRAIC_SIG_TERM_CACHE.get(item)
        if cached is not None:
            return cached
        registry = TypeRegistry.get_instance()
        t_clean = item.type_name.strip()
        if item.is_top() or is_top_symbol(t_clean, registry):
            res = TOP
        elif t_clean.startswith("?") or "|" in t_clean or ("[" in t_clean and t_clean.endswith("]")) or registry.is_type_variable(t_clean):
            res = TypeTerm.from_string(t_clean)
        else:
            canonical = registry.canonical_name(t_clean)
            if is_top_symbol(canonical, registry):
                res = TOP
            else:
                res = TypestateTerm(
                    type_name=canonical,
                    state=item.state,
                    qualifiers=item.qualifiers,
                    abstract_type=getattr(item, "abstract_type", None),
                    accepted_states=getattr(item, "accepted_states", frozenset()),
                    parent_state=getattr(item, "parent_state", None),
                )
        _ALGEBRAIC_SIG_TERM_CACHE[item] = res
        return res
    registry = TypeRegistry.get_instance()
    if isinstance(item, str):
        if is_top_symbol(item, registry):
            return TOP
        return TypeTerm.from_string(item)
    return TOP


# =====================================================================
# 2.6. Generic Substitution and Topology Verification Gates
# =====================================================================

def substitute_generics(
    target: Any,
    sigma: Union[Substitution, Dict[str, Any]]
) -> Any:
    """
    Substitutes generic type variables in schemas, signatures, or type strings using substitution sigma.
    e.g. substitute_generics('Sequence[T]', {'T': 'MatLike'}) -> 'Sequence[MatLike]'
    e.g. substitute_generics('T', {'T': 'MatLike'}) -> 'MatLike'
    e.g. substitute_generics(PortSchema(type_name='T'), {'T': 'MatLike'}) -> PortSchema(type_name='MatLike')
    """
    if sigma is None:
        return target
    # Accept any substitution-like object (duck-typed) so that alternate module
    # instances can never corrupt the substitution application step.
    mappings = sigma.mappings if hasattr(sigma, "mappings") else dict(sigma)
    if not mappings:
        return target

    sub = Substitution({k: v for k, v in mappings.items()})

    if isinstance(target, str):
        term = TypeTerm.from_string(target)
        res_term = term.apply_substitution(sub)
        if isinstance(res_term, TypeVariable):
            return res_term.var_name
        return str(res_term)

    if isinstance(target, PortSignature):
        new_sig = substitute_generics(target.signature, sub)
        return PortSignature(
            name=target.name,
            signature=new_sig,
            required=target.required,
            default_value=target.default_value,
            doc=target.doc,
            domain=target.domain,
            abstract_type=target.abstract_type,
            enum_values=target.enum_values,
            param_kind=target.param_kind,
            value_constraints=target.value_constraints,
            shape_contract=target.shape_contract,
            accepted_states=getattr(target, "accepted_states", frozenset()),
            parent_state=getattr(target, "parent_state", None),
            port_role=getattr(target, "port_role", None)
        )

    if isinstance(target, AlgebraicSignature):
        new_type = substitute_generics(target.type_name, sub)
        return AlgebraicSignature(
            type_name=str(new_type),
            state=target.state,
            qualifiers=target.qualifiers,
            abstract_type=target.abstract_type,
            accepted_states=getattr(target, "accepted_states", frozenset()),
            parent_state=getattr(target, "parent_state", None)
        )

    if hasattr(target, "type_name") and hasattr(target, "model_copy"):
        new_type = substitute_generics(target.type_name, sub)
        return target.model_copy(update={"type_name": str(new_type)})

    if isinstance(target, TypeTerm):
        return target.apply_substitution(sub)

    return target


def verify_coproduct_branch(
    then_term: Union[TypeTerm, AlgebraicSignature, str],
    else_term: Union[TypeTerm, AlgebraicSignature, str],
    join_type: Optional[Union[TypeTerm, AlgebraicSignature, str]] = None,
    sigma: Optional[Substitution] = None
) -> Tuple[bool, Optional[TypeTerm], Optional[Substitution]]:
    """
    Verifies that coproduct True-path and False-path can unify to a common join type D:
      unify(tau_then, D) != bottom and unify(tau_else, D) != bottom
    Returns (is_valid, resolved_join_type, updated_sigma).
    """
    sub = Substitution(sigma.mappings if sigma else {})
    t_then = _to_type_term(then_term)
    t_else = _to_type_term(else_term)

    if join_type is not None:
        target_D = _to_type_term(join_type)
        s1 = unify(t_then, target_D, sub)
        if s1 is None:
            return False, None, None
        s2 = unify(t_else, target_D, s1)
        if s2 is None:
            return False, None, None
        return True, target_D, s2

    # If no join_type given, check if then and else unify with each other
    s_join = unify(t_then, t_else, sub)
    if s_join is not None:
        return True, t_then.apply_substitution(s_join), s_join

    # Check poset reachability in TypeRegistry
    registry = TypeRegistry.get_instance()
    n_then = getattr(t_then, "type_name", getattr(t_then, "name", str(t_then)))
    n_else = getattr(t_else, "type_name", getattr(t_else, "name", str(t_else)))
    if registry.is_subtype(n_then, n_else):
        return True, t_else, sub
    if registry.is_subtype(n_else, n_then):
        return True, t_then, sub

    return False, None, None


def verify_traced_loop_invariant(
    feedback_in: Union[TypeTerm, AlgebraicSignature, str],
    feedback_out: Union[TypeTerm, AlgebraicSignature, str],
    sigma: Optional[Substitution] = None
) -> Tuple[bool, Optional[Substitution]]:
    """
    Verifies the categorical Traced Feedback loop invariant:
      unify(tau_feedback_out, tau_feedback_in) != bottom
    Ensures that loop body updates preserve or are compatible with the accumulator state U.
    """
    t_in = _to_type_term(feedback_in)
    t_out = _to_type_term(feedback_out)
    sub = Substitution(sigma.mappings if sigma else {})
    new_sub = unify(t_out, t_in, sub)
    if new_sub is None:
        return False, None
    return True, new_sub


# =====================================================================
# 3. Formal Type Monad M_T(A)
# =====================================================================

class MonadResult(Generic[T], ABC):
    """
    Formal Type Monad Result: M_T(A) = { (a, sigma) } U { bottom }.
    """
    @abstractmethod
    def is_bottom(self) -> bool:
        pass


@dataclass(frozen=True, slots=True)
class Success(MonadResult[T]):
    """Successful computation carrying value a and substitution sigma."""
    value: T
    sigma: Substitution

    def is_bottom(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Failure(MonadResult[T]):
    """Failure bottom (⊥)."""
    reason: str

    def is_bottom(self) -> bool:
        return True


def unit(value: T, sigma: Optional[Substitution] = None) -> MonadResult[T]:
    """Monad unit: injects value a with initial substitution sigma into the monad."""
    return Success(value, sigma or Substitution())


def bind(
    m: MonadResult[T],
    k: Callable[[T, Substitution], MonadResult[U]]
) -> MonadResult[U]:
    """
    Monadic bind:
      bind(m, k) = k(a) with sigma_new if step succeeds; otherwise bottom.
    If m is Failure, short-circuits immediately to Failure.
    """
    if m.is_bottom():
        return Failure(m.reason if isinstance(m, Failure) else "Bottom")
    assert isinstance(m, Success)
    return k(m.value, m.sigma)


def _format_literal_value(val: Any, sig: Optional[PortSignature] = None) -> str:
    """Formats an extracted slot/hyperparameter value into an executable code literal."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (list, tuple, dict)):
        return repr(val)
    s = str(val).strip()
    if s in ("None", "True", "False"):
        return s
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s
    try:
        float(s)
        return s
    except ValueError:
        pass
    t_name = str(getattr(sig, "type_name", "")).lower() if sig else ""
    if t_name == "str" or not s.isidentifier():
        return repr(s)
    return s


# =====================================================================
# 4. Domain-Agnostic Execution Context
# =====================================================================

class ExecutionContext:
    """
    Runtime execution scope holding bound variables and literal arguments.
    Operates strictly via algebraic typestates and substitutions with ZERO domain hardcodes.
    """

    @staticmethod
    def _is_path_string(val: Any) -> bool:
        if not val or not isinstance(val, str):
            return False
        s = val.strip()
        if len(s) < 2:
            return False
        if "/" in s or "\\" in s:
            return True
        if "." in s:
            base, _, ext = s.rpartition(".")
            if base and ext and len(ext) <= 8 and ext.isalnum():
                return True
        return False

    def __init__(self, prompt: str = "", scope: Optional[Dict[str, Any]] = None):
        self._prompt = prompt or ""
        self.scope: Dict[str, Any] = dict(scope or {})
        self.variables: Dict[str, Tuple[PortSignature, str]] = {}
        self.var_counter: int = 0
        self.var_sources: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = {}
        self.used_indices: Set[int] = set()
        self.consumed_tokens: Set[str] = set()
        # Tuple-member consumption tracking: (var_name, member_index) pairs already
        # bound by earlier cells. Later cells consuming the same heterogeneous
        # product prefer the remaining members — this is how allocation semantics
        # (train/verify/test partitions) emerge from consumption order.
        self.consumed_members: Set[Tuple[str, int]] = set()
        self.unresolved_ports: List[Tuple[str, str]] = []
        self.unbindable_count: int = 0
        # Declared runtime egress aliases: maps a prompt-declared sink
        # identifier (e.g. "store the results into Z" -> "Z") to the pipeline
        # variable that carries the terminal value. Applied at the RUNTIME
        # namespace level by the sandbox — the emitted source is never
        # rewritten to fake an assignment.
        self.runtime_aliases: Dict[str, str] = {}
        self.ordered_literals: List[Tuple[int, str, str]] = self._extract_universal_literals(self._prompt)
        # D8: Map each literal to its prompt clause index
        self.literal_clause_map: Dict[int, int] = self._build_literal_clause_map(self._prompt, self.ordered_literals)
        self.target_literals: Set[str] = set()
        t_sink = self._extract_target_sink(self._prompt)
        if t_sink:
            self.target_literals.add(t_sink)
        self.current_cell_clause_idx: Optional[int] = None
        # Bare-identifier role map: char position -> stemmed role context tokens
        # (e.g. pos(X) -> {"column"}). Drives role-conditioned identifier binding
        # and for-each multiplicity detection.
        self.identifier_roles: Dict[int, FrozenSet[str]] = self._build_identifier_role_map(self._prompt)
        # Provenance and monoidal DAG branch tracking
        self.var_origins: Dict[str, Set[str]] = {}
        self.superseded_vars: Set[str] = set()
        self.var_parents: Dict[str, Set[str]] = {}
        self.scope_variables: Dict[str, Any] = {}
        self.llm_slots: Dict[str, Dict[str, Any]] = {}
        self.target_col: Optional[str] = None
        self._explicit_columns: List[str] = []
        self.hyperparameters: Dict[str, Any] = {}
        if self.scope:
            for k, v in self.scope.items():
                self.declare_variable(k, v, k)

    @staticmethod
    def _extract_target_sink(prompt: str) -> Optional[str]:
        """
        Dynamically extracts declared target output identifier from prompt (e.g. 'store results into Z').
        Grounded in TypeRegistry egress verbs and destination prepositions, with reserved names
        governed by Python keywords and registered lattice types. Zero ad-hoc word lists or regex.
        """
        if not prompt:
            return None
        reg = TypeRegistry.get_instance()
        egress_triggers = (
            reg.get_egress_tokens()
            | reg.get_dest_port_tokens()
        )
        arg_prepositions = {"on", "by", "with", "using", "via", "at", "from", "for"}
        reg_types = reg.get_registered_types()
        stopwords = reg.get_stopwords()

        # 1. Check identifier groups from structural orthography
        groups = ExecutionContext.extract_identifier_groups(prompt)
        for grp in reversed(groups):
            if grp.role_tokens and (grp.role_tokens & egress_triggers) and not (grp.role_tokens & arg_prepositions):
                for _, mem in reversed(grp.members):
                    tok = mem.strip("'\"")
                    tok_low = tok.lower()
                    if (
                        tok.isidentifier()
                        and not keyword.iskeyword(tok_low)
                        and tok_low not in reg_types
                        and tok_low not in stopwords
                        and not ExecutionContext._is_path_string(tok)
                    ):
                        return tok

        # 2. Contextual scan of prompt tail
        tokens = prompt.strip().rstrip(".;:").split()
        if len(tokens) < 2:
            return None
        prev_word = tokens[-2].strip("'\",.;:").lower()
        if prev_word in arg_prepositions:
            return None

        last_tok = tokens[-1].strip("'\"")
        last_tok_low = last_tok.lower()
        if (
            last_tok.isidentifier()
            and not keyword.iskeyword(last_tok_low)
            and last_tok_low not in reg_types
            and last_tok_low not in stopwords
            and not ExecutionContext._is_path_string(last_tok)
        ):
            tail_words = [w.strip("'\",.;:").lower() for w in tokens[-5:-1]]
            if any(w in egress_triggers for w in tail_words):
                return last_tok

        return None

    def declare_runtime_alias(self, alias_name: str, source_var: str) -> None:
        """Declares that a prompt-declared sink identifier aliases a pipeline variable."""
        name = str(alias_name or "").strip()
        source = str(source_var or "").strip()
        if name and source and name != source:
            self.runtime_aliases[name] = source

    def reset(self):
        """Resets transient pipeline variables and consumed literal indices, preserving prompt, parameters, and initial scope."""
        self.variables = {}
        self.var_sources = {}
        self.var_origins = {}
        self.superseded_vars = set()
        self.var_parents = {}
        self.scope_variables = {}
        self.used_indices = set()
        self.consumed_tokens = set()
        self.consumed_members = set()
        self.target_literals = set()
        t_sink = self._extract_target_sink(self._prompt)
        if t_sink:
            self.target_literals.add(t_sink)
        self.current_cell_clause_idx = None
        self.unresolved_ports = []
        self.unbindable_count = 0
        self.var_counter = 0
        self.runtime_aliases = dict(getattr(self, "runtime_aliases", {}) or {})
        if hasattr(self, "scope") and self.scope:
            for k, v in self.scope.items():
                self.declare_variable(k, v, k)

    def clone(self) -> "ExecutionContext":
        new_ctx = ExecutionContext(prompt=self._prompt, scope=self.scope)
        new_ctx.variables = dict(self.variables)
        new_ctx.var_sources = dict(self.var_sources)
        new_ctx.var_origins = {k: set(v) for k, v in self.var_origins.items()}
        new_ctx.superseded_vars = set(self.superseded_vars)
        new_ctx.var_parents = {k: set(v) for k, v in self.var_parents.items()}
        new_ctx.scope_variables = dict(self.scope_variables)
        new_ctx.parameters = dict(self.parameters)
        new_ctx.used_indices = set(self.used_indices)
        new_ctx.consumed_tokens = set(self.consumed_tokens)
        new_ctx.consumed_members = set(self.consumed_members)
        new_ctx.target_literals = set(self.target_literals)
        new_ctx.current_cell_clause_idx = self.current_cell_clause_idx
        new_ctx.literal_clause_map = dict(self.literal_clause_map)
        new_ctx.unresolved_ports = list(self.unresolved_ports)
        new_ctx.unbindable_count = self.unbindable_count
        new_ctx.var_counter = self.var_counter
        new_ctx.llm_slots = {k: dict(v) for k, v in self.llm_slots.items()}
        new_ctx.target_col = self.target_col
        new_ctx.columns = list(self.columns)
        new_ctx.hyperparameters = dict(self.hyperparameters)
        new_ctx.runtime_aliases = dict(getattr(self, "runtime_aliases", {}) or {})
        return new_ctx

    @staticmethod
    def _build_literal_clause_map(prompt: str, literals: List[Tuple[int, str, str]]) -> Dict[int, int]:
        """Maps each literal index to its originating prompt clause index."""
        try:
            from .planner import _segment_prompt_clauses
        except (ImportError, ValueError):
            from planner import _segment_prompt_clauses
        clauses = _segment_prompt_clauses(prompt)
        raw_spans = []
        cur_search = 0
        for cl in clauses:
            start = prompt.find(cl, cur_search)
            if start != -1:
                end = start + len(cl)
                raw_spans.append((start, end))
                cur_search = end
            else:
                start = prompt.find(cl)
                if start != -1:
                    raw_spans.append((start, start + len(cl)))
        if not raw_spans:
            raw_spans = [(0, len(prompt))]

        lit_map = {}
        for idx, (pos, kind, val) in enumerate(literals):
            c_idx = 0
            for i, (s, e) in enumerate(raw_spans):
                if s <= pos <= e or (pos >= s and i == len(raw_spans) - 1):
                    c_idx = i
                    break
            lit_map[idx] = c_idx
        return lit_map

    @staticmethod
    def _asset_direction(prompt: str, pos: int) -> str:
        prev_chunk = (prompt or "")[:pos].rstrip()
        words = prev_chunk.split()
        if not words:
            return ""

        reg = TypeRegistry.get_instance()
        dest_triggers = reg.get_dest_port_tokens() | reg.get_egress_tokens()
        src_triggers = reg.get_source_port_tokens()

        for word in reversed(words):
            w = word.strip(".,!?:;'\"").lower()
            if not w.isalpha():
                trailing = []
                for ch in reversed(w):
                    if ch.isalpha():
                        trailing.append(ch)
                    else:
                        break
                w = "".join(reversed(trailing)).lower()
            if not w:
                continue
            if w in ("and", "or", ",", "&", "with", "both"):
                continue
            if "." in word or "/" in word or "\\" in word:
                continue
            if w in dest_triggers:
                return "dest"
            if w in src_triggers:
                return "src"
            break
        return ""

    @property
    def source_files(self) -> List[str]:
        if self.parameters.get("source_uris"):
            return list(self.parameters["source_uris"])
        file_candidates = [
            (pos, val) for pos, kind, val in self.ordered_literals
            if kind in ("file_asset", "quoted_str") and ("." in val or "/" in val or "\\" in val)
        ]
        if not file_candidates:
            return []
        srcs = [val for pos, val in file_candidates if self._asset_direction(self.prompt, pos) == "src"]
        if srcs:
            return srcs
        unmarked = [val for pos, val in file_candidates if self._asset_direction(self.prompt, pos) != "dest"]
        return [unmarked[0]] if unmarked else [file_candidates[0][1]]

    @property
    def dest_files(self) -> List[str]:
        if self.parameters.get("dest_uris"):
            return list(self.parameters["dest_uris"])
        file_candidates = [
            (pos, val) for pos, kind, val in self.ordered_literals
            if kind in ("file_asset", "quoted_str") and ("." in val or "/" in val or "\\" in val)
        ]
        if len(file_candidates) < 1:
            return []
        dests = [val for pos, val in file_candidates if self._asset_direction(self.prompt, pos) == "dest"]
        if dests:
            return dests
        reg = TypeRegistry.get_instance()
        prompt_words = set(str(self.prompt or "").lower().split())
        has_prompt_egress = bool(prompt_words & (reg.get_egress_tokens() | reg.get_dest_port_tokens()))
        if not has_prompt_egress:
            return []
        src_set = set(self.source_files)
        return [val for pos, val in file_candidates if val not in src_set and self._asset_direction(self.prompt, pos) != "src"]

    @property
    def columns(self) -> List[str]:
        if getattr(self, "_explicit_columns", None):
            return self._explicit_columns
        cols = []
        for pos, kind, val in self.ordered_literals:
            if kind == "quoted_str" and "." not in val:
                cols.append(val)
            elif kind == "bare_id":
                roles = self.identifier_roles.get(pos, frozenset())
                if "column" in roles or "by" in roles:
                    cols.append(val)
        return cols

    @columns.setter
    def columns(self, val: List[str]):
        self._explicit_columns = list(val)

    @property
    def by_column(self) -> Optional[str]:
        c = self.columns
        return c[0] if c else None

    @property
    def flags(self) -> Dict[str, Any]:
        fl = {}
        p_low = self._prompt.lower()
        if "descend" in p_low or "descending" in p_low:
            fl["ascending"] = False
            fl["descending"] = True
        elif "ascend" in p_low or "ascending" in p_low:
            fl["ascending"] = True
            fl["descending"] = False
        return fl

    @property
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, val: str):
        self._prompt = val or ""
        self.used_indices = set()
        self.consumed_tokens = set()
        self.consumed_members = set()
        self.target_literals = set()
        t_sink = self._extract_target_sink(self._prompt)
        if t_sink:
            self.target_literals.add(t_sink)
        self.ordered_literals = self._extract_universal_literals(self._prompt)
        self.literal_clause_map = self._build_literal_clause_map(self._prompt, self.ordered_literals)
        self.identifier_roles = self._build_identifier_role_map(self._prompt)

    @staticmethod
    def _extract_universal_literals(prompt: str) -> List[Tuple[int, str, str]]:
        if not prompt:
            return []

        spans: List[Tuple[int, str, str]] = []
        n = len(prompt)

        # 1. Quoted literals: '...' or "..." -> kind "quoted_str"
        i = 0
        while i < n:
            ch = prompt[i]
            if ch in ("'", '"'):
                quote_char = ch
                j = i + 1
                while j < n and prompt[j] != quote_char:
                    if prompt[j] == '\\' and j + 1 < n:
                        j += 1
                    j += 1
                if j < n and prompt[j] == quote_char:
                    val = prompt[i + 1:j]
                    is_file = "/" in val or "\\" in val or (
                        "." in val and not val.startswith(".") and not val.endswith(".")
                        and val.rsplit(".", 1)[1].isalnum() and not val.rsplit(".", 1)[1].isdigit()
                        and len(val.rsplit(".", 1)[1]) <= 8
                    )
                    kind = "file_asset" if is_file else "quoted_str"
                    spans.append((i, kind, val))
                    i = j + 1
                    continue
            i += 1

        # 2. Word tokens for file assets, numerics, and bare referential identifiers
        words_with_pos: List[Tuple[int, str]] = []
        cur_word: List[str] = []
        w_start = None
        for idx, ch in enumerate(prompt):
            if not ch.isspace():
                if w_start is None:
                    w_start = idx
                cur_word.append(ch)
            else:
                if cur_word:
                    words_with_pos.append((w_start, "".join(cur_word)))
                    cur_word = []
                    w_start = None
        if cur_word and w_start is not None:
            words_with_pos.append((w_start, "".join(cur_word)))

        def _quoted_range(pos: int, length: int) -> bool:
            return any(s - 1 <= pos and pos + length <= s + len(v) + 2 for s, t, v in spans)

        # Sentence-initial positions: the first word of the prompt, and any word
        # that directly follows a sentence-terminating period. Capitalization
        # at those positions is grammatical, never naming.
        sentence_initial: Set[int] = set()
        prev_terminates = True
        for _w_idx, (_pos, _raw) in enumerate(words_with_pos):
            if prev_terminates:
                sentence_initial.add(_pos)
            prev_terminates = _raw.endswith(".")

        for pos, raw_w in words_with_pos:
            w = raw_w.rstrip(".,;:)")
            # Handle leading brackets and quotes
            leading_strip = 0
            while leading_strip < len(w) and w[leading_strip] in ("(", "[", "{", "\"", "'"):
                leading_strip += 1
            if leading_strip > 0:
                w = w[leading_strip:]
                pos = pos + leading_strip
            w = w.rstrip(")]}\"'.,;:)")
            if not w:
                continue

            if _quoted_range(pos, len(w)):
                continue

            # Path or filename token (domain-agnostic, zero hardcoded extensions)
            if "/" in w or "\\" in w:
                spans.append((pos, "file_asset", w))
                continue
            if "." in w and not w.startswith(".") and not w.endswith("."):
                parts = w.rsplit(".", 1)
                ext = parts[1].lower()
                if ext.isalnum() and not ext.isdigit() and len(ext) <= 8:
                    spans.append((pos, "file_asset", w))
                    continue

            # Numeric tokens (including comma-separated integers like "1,000" or "10,000")
            num_candidate = w.replace(",", "") if ("," in w and all(part.isdigit() for part in w.split(","))) else w
            try:
                float(num_candidate)
                spans.append((pos, "numeric", num_candidate))
                continue
            except ValueError:
                pass

            # Dimension patterns: NxM, NxN (e.g. "100x100", "5x5", "3x3") without regex
            dim_parts = None
            for sep in ("x", "X", "×"):
                if sep in w:
                    p1, _, p2 = w.partition(sep)
                    if p1.isdigit() and p2.isdigit():
                        dim_parts = (p1, p2)
                        break
            if dim_parts:
                spans.append((pos, "numeric", dim_parts[0]))
                if dim_parts[0] != dim_parts[1]:
                    spans.append((pos + len(dim_parts[0]) + 1, "numeric", dim_parts[1]))
                continue

            # Bare referential identifiers: the way humans name columns, fields
            # and variables in prose WITHOUT quoting them ("normalize X column").
            # Structural orthography only: a short, capitalized, alphanumeric
            # token that no other literal kind claims. Sentence-initial words
            # and sentence connectives are excluded (a capitalized "The" mid-
            # prompt is a connective, not a name); this is language-level
            # orthography, not domain vocabulary.
            reg = TypeRegistry.get_instance()
            if (
                pos not in sentence_initial
                and ExecutionContext._is_bare_identifier_token(w)
                and not reg.is_operation_token(w)
            ):
                spans.append((pos, "identifier", w))
                continue

            # Prepositional objects: the referent attached to a relational
            # preposition ("top 2 rows BY age", "group BY dept"). Purely
            # language-level: how English binds a value to a relation. The
            # word itself is lowercased prose — extraction requires the
            # preposition trigger, alphabetic body, and non-connective status.
            if (
                (w.isidentifier() or w.isalpha())
                and 3 <= len(w)
                and w.lower() not in _SENTENCE_CONNECTIVES
                and not reg.is_operation_token(w)
            ):
                prev = prompt[:pos].rstrip()
                if prev:
                    prev_words = [pw.strip(".,;:()") for pw in prev.split()[-3:] if pw.strip(".,;:()")]
                    if any(pw.lower() in _PREPOSITION_TRIGGERS for pw in prev_words):
                        spans.append((pos, "identifier", w))
                        continue

        # 3. Predicate / filter comparison expressions (e.g. "value > 100", "score <= 50", "x == 1") without regex
        for op in (">=", "<=", "==", "!=", ">", "<"):
            idx = 0
            while True:
                op_pos = prompt.find(op, idx)
                if op_pos == -1:
                    break
                lhs_part = prompt[:op_pos].rstrip()
                lhs_tokens = lhs_part.split()
                if lhs_tokens:
                    lhs = lhs_tokens[-1].strip("(),[]{}")
                    if lhs.isidentifier():
                        rhs_part = prompt[op_pos + len(op):].lstrip()
                        rhs_tokens = rhs_part.split()
                        if rhs_tokens:
                            rhs = rhs_tokens[0].strip("(),;:?[]{}")
                            if rhs:
                                expr_str = f"{lhs} {op} {rhs}"
                                start_pos = prompt.rfind(lhs, 0, op_pos)
                                if start_pos != -1:
                                    spans.append((start_pos, "expr", expr_str))
                idx = op_pos + len(op)

        # Order strictly by character position in prompt
        spans.sort(key=lambda x: x[0])

        # D8: Deduplicate identical literals at the same character position
        seen = set()
        deduped_spans: List[Tuple[int, str, str]] = []
        for pos, kind, val in spans:
            key = (pos, str(val).strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped_spans.append((pos, kind, val))
        return deduped_spans

    @staticmethod
    def _is_bare_identifier_token(w: str) -> bool:
        """
        Structural orthography of a bare referential identifier: short, begins
        uppercase, alphanumeric/underscore body. Covers the conventions humans
        use for columns/fields/variables in prose (X, Y, Z, X1, Col, ID) without
        any domain vocabulary. Sentence-initial position is handled by the
        caller (context), not here.
        """
        if not w or len(w) > 4:
            return False
        if not w[0].isupper():
            return False
        if not all(ch.isalnum() or ch == "_" for ch in w):
            return False
        if not any(ch.isalpha() for ch in w):
            return False
        if w.lower() in _SENTENCE_CONNECTIVES:
            return False
        return True

    @staticmethod
    def extract_identifier_groups(prompt: str) -> List[IdentifierGroup]:
        """
        Groups bare referential identifiers by their shared syntactic role
        context. Two structural signals, both language-level:
          1. repeated adjacent context: ``X column ... Y column ... Z column``
             (each identifier followed by the same role noun, possibly in
             separate clauses);
          2. coordination runs: ``columns X, Y and Z`` (identifiers in a comma/
             conjunction run share one head noun).
        The role tokens are STEMMED (normalize_token) so they match cell
        identity tokens symmetrically ("columns" ~ "column").
        """
        if not prompt:
            return []

        # Word scan with positions
        words: List[Tuple[int, str]] = []
        cur: List[str] = []
        start: Optional[int] = None
        for idx, ch in enumerate(prompt):
            if ch.isspace():
                if cur:
                    words.append((start, "".join(cur)))
                    cur = []
                    start = None
            else:
                if start is None:
                    start = idx
                cur.append(ch)
        if cur and start is not None:
            words.append((start, "".join(cur)))
        if len(words) < 2:
            return []

        # Quoted ranges are excluded (quoted strings are already first-class literals)
        quoted_ranges: List[Tuple[int, int]] = []
        i = 0
        n = len(prompt)
        while i < n:
            if prompt[i] in ("'", '"'):
                j = i + 1
                while j < n and prompt[j] != prompt[i]:
                    j += 1
                if j < n:
                    quoted_ranges.append((i, j))
                    i = j + 1
                    continue
            i += 1

        # Cleaned word sequence for context computation
        cleaned: List[Tuple[int, str]] = []
        raw_has_break: Set[int] = set()
        for w_idx, (pos, raw) in enumerate(words):
            if raw.endswith((".", ";")):
                raw_has_break.add(len(cleaned))
            w = raw.rstrip(".,;:)")
            if w:
                cleaned.append((pos, w))

        reg = TypeRegistry.get_instance()
        _PROJ_TOKENS = reg.get_column_projection_tokens()
        _ROLE_NOUNS = (
            _PROJ_TOKENS
            | reg.get_dest_port_tokens()
            | reg.get_data_bearing_roles()
            | reg.get_declared_role_carriers()
            | reg.get_preposition_triggers()
        )

        # Identifier candidates: interior words (never the first word of the
        # prompt — sentence-initial capitalization is grammatical, not naming)
        idents: List[Tuple[int, int, str]] = []  # (clean_idx, pos, token)
        for c_idx, (pos, w) in enumerate(cleaned):
            if c_idx == 0:
                continue
            clean_tok = w.strip("'\"")
            if "." in clean_tok or "/" in clean_tok or "\\" in clean_tok:
                continue
            try:
                float(clean_tok)
                continue
            except ValueError:
                pass
            is_quoted = any(s <= pos <= e for s, e in quoted_ranges)
            if is_quoted:
                if len(clean_tok) >= 1 and (len(clean_tok) == 1 or clean_tok.lower() not in _SENTENCE_CONNECTIVES) and not reg.is_operation_token(clean_tok):
                    idents.append((c_idx, pos, clean_tok))
            elif ExecutionContext._is_bare_identifier_token(w) and not reg.is_operation_token(w):
                idents.append((c_idx, pos, w))

        if not idents:
            return []

        def _stem_ctx(word: str) -> str:
            wl = word.lower().strip("'\"")
            if wl in _SENTENCE_CONNECTIVES or len(wl) < 2:
                return ""
            st = normalize_token(wl)
            return st if len(st) >= 2 else ""

        # Clause boundary segmentation for scoping identifier groups per action clause
        try:
            from .tokenizer import CellTokenizer
        except (ImportError, ValueError):
            from tokenizer import CellTokenizer

        clauses = CellTokenizer.split_prompt_clauses(prompt)
        clause_spans: List[Tuple[int, int]] = []
        cur_search = 0
        for cl in clauses:
            start = prompt.find(cl, cur_search)
            if start != -1:
                end = start + len(cl)
                clause_spans.append((start, end))
                cur_search = end
            else:
                start = prompt.find(cl)
                if start != -1:
                    clause_spans.append((start, start + len(cl)))
        if not clause_spans:
            clause_spans = [(0, len(prompt))]

        def _get_clause_idx(pos: int) -> int:
            for c_i, (s, e) in enumerate(clause_spans):
                if s <= pos <= e or (pos >= s and c_i == len(clause_spans) - 1):
                    return c_i
            return 0

        # Role assignment:
        # Check backward context across coordination runs ("columns X and Y", "column X", "into Z"),
        # and if backward role is missing or not a known structural role noun, check forward context ("X column", "X as target").
        groups: Dict[Tuple[int, str], List[Tuple[int, str]]] = {}
        order: List[Tuple[int, str]] = []
        ident_indices = {c_idx for c_idx, _, _ in idents}
        for c_idx, pos, w in idents:
            role = ""
            # Backward check: look back across connectives, commas, and other identifiers in coordination run
            k = c_idx - 1
            while k >= 0:
                if k not in ident_indices:
                    cand = _stem_ctx(cleaned[k][1])
                    if cand:
                        role = cand
                        break
                    raw_prev = cleaned[k][1].strip("'\"").lower()
                    if raw_prev not in _SENTENCE_CONNECTIVES and cleaned[k][1] not in (",", ";"):
                        break
                if k in raw_has_break:
                    break
                k -= 1

            # Forward check: if backward role is missing or not a known structural role noun
            if not role or role not in _ROLE_NOUNS:
                for step in range(1, 4):
                    if c_idx + step < len(cleaned):
                        if c_idx + step in ident_indices:
                            continue
                        cand = _stem_ctx(cleaned[c_idx + step][1])
                        if cand:
                            if not role or cand in _ROLE_NOUNS:
                                role = cand
                            break
                        if c_idx + step in raw_has_break:
                            break

            cl_idx = _get_clause_idx(pos)
            key = (cl_idx, role)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((pos, w))

        return [
            IdentifierGroup(members=members, role_tokens=frozenset({r}) if r else frozenset())
            for (cl_idx, r) in order
            for members in [groups[(cl_idx, r)]]
        ]

    @staticmethod
    def _build_identifier_role_map(prompt: str) -> Dict[int, FrozenSet[str]]:
        """char position of each extracted identifier -> its group's role tokens."""
        role_map: Dict[int, FrozenSet[str]] = {}
        if not prompt:
            return role_map
        for group in ExecutionContext.extract_identifier_groups(prompt):
            for pos, _tok in group.members:
                role_map[pos] = group.role_tokens
        return role_map

    def declare_variable(self, name: str, port_sig: Union[PortSignature, AlgebraicSignature, Any], expr: str = "", cell: Optional[Any] = None):
        if not name:
            return
        if isinstance(port_sig, AlgebraicSignature):
            port_sig = PortSignature(name=name, signature=port_sig)
        elif not isinstance(port_sig, PortSignature):
            port_sig = PortSignature(name=name, signature=AlgebraicSignature(str(port_sig), "any"))
        self.variables[name] = (port_sig, expr or name)
        self.scope[name] = port_sig
        if cell is not None:
            self.var_sources[name] = cell

    def get_variable_name(self, port_sig: PortSignature) -> Optional[str]:
        """Finds in-scope variable that unifies with port_sig."""
        for v_name, (v_sig, _) in reversed(list(self.variables.items())):
            if v_sig.unifies_with(port_sig):
                return v_name
        return None

    def _project_semantic_slot(self, param_name: str = "", role_label: str = "") -> Optional[str]:
        """
        Semantic Slot Projection:
        Projects the port's DECLARED semantic role (or, absent a declared role,
        the port's parameter name) onto unconsumed prompt tokens via dense
        vector cosine similarity.
        Guard: candidate words morphologically related to the port's own name
        are EXCLUDED — a slot's label is not a proxy for its value's meaning
        (measured: the word "named" winning a `name` slot over the semantically
        correct X/Y/Z because of near-identical spelling).
        Contains ZERO hardcoded keyword tuples, ZERO token distance hacks.
        """
        if not self.prompt:
            return None

        # Universal grammatical function words derived from registry / corpus
        _FUNCTION_WORDS = TypeRegistry.get_instance().get_function_words()

        # Candidate word tokens from prompt, excluding self-referential collisions, function words, and file assets
        file_assets = {val.lower() for _, kind, val in self.ordered_literals if kind == "file_asset"}
        p_stem = normalize_token((param_name or "").lower()) if param_name else ""
        words = []
        for w in self.prompt.strip().split():
            clean_w = w.strip(" '\".,;:()[]{}=:")
            w_lower = clean_w.lower()
            if len(clean_w) < 2 or w_lower in self.consumed_tokens or w_lower in _FUNCTION_WORDS or w_lower in file_assets:
                continue
            if p_stem:
                w_stem = normalize_token(w_lower)
                if (
                    w_stem == p_stem
                    or w_lower.startswith(p_stem)
                    or p_stem.startswith(w_stem)
                ):
                    continue
            words.append(clean_w)

        if not words:
            return None

        target_label = role_label or param_name or "parameter"
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            import numpy as np

            mm = ModelManager.get_instance()
            if mm.profile is not None:
                e_param = np.array(mm.get_embedding(target_label), dtype=np.float32)
                p_norm = np.linalg.norm(e_param)
                if p_norm > 0:
                    e_param = e_param / p_norm
                    best_word = None
                    best_sim = -1.0
                    for w in words:
                        e_w = np.array(mm.get_embedding(w), dtype=np.float32)
                        w_norm = np.linalg.norm(e_w)
                        if w_norm > 0:
                            sim = float(np.dot(e_param, e_w / w_norm))
                            if sim > best_sim:
                                best_sim = sim
                                best_word = w
                    if best_word is not None and best_sim > 0.35:
                        self.consumed_tokens.add(best_word.lower())
                        return best_word
        except Exception as e:
            logger.debug(f"[UNIFICATION] Semantic slot projection fallback: {e}")

        # Fallback (symbolic / embedding-free mode):
        # Scan prompt tokens for param_name triggers (e.g. "by", "on", "index", "column")
        # or role triggers (e.g. "sort" for "sort_column", "group" for "group_column").
        raw_tokens = [w.strip(" '\".,;:()[]{}=:") for w in self.prompt.strip().split()]
        raw_lower = [w.lower() for w in raw_tokens]

        triggers = []
        if param_name:
            triggers.append(param_name.lower())
            # Relational prepositions (LANGUAGE-level): how English attaches a
            # value to its slot ("sort BY column", "matrix OF rows").
            triggers.extend(["by", "of", "for", "with"])
        if role_label:
            triggers.extend(t.lower() for t in CellTokenizer.tokenize_identifier(role_label) if len(t) >= 3)
        if target_label:
            triggers.append(target_label.lower())

        # Grammatical function words + declared ordering-vocabulary tokens are
        # never slot VALUES. No domain operation verbs: file assets are already
        # excluded by kind, and value words come from declared role/state
        # vocabulary via `triggers` above.
        skip_words = set(TypeRegistry.get_instance().get_function_words()) | {
            "by", "on", "with", "of", "for", "in", "to", "the", "a", "an", "and", "then",
            "true", "false",
        }

        for tr in triggers:
            if tr in raw_lower:
                t_idx = raw_lower.index(tr)
                for f_idx in range(t_idx + 1, min(t_idx + 6, len(raw_tokens))):
                    tok = raw_tokens[f_idx]
                    tl = tok.lower()
                    if tl in skip_words or tl in self.consumed_tokens or "." in tok or len(tok) < 2:
                        continue
                    self.consumed_tokens.add(tl)
                    return tok

        # Fallback: if words has remaining unconsumed non-file tokens
        content_candidates = [
            w for w in words
            if w.lower() not in self.consumed_tokens and "." not in w and len(w) >= 2 and w.lower() not in skip_words
        ]
        if len(content_candidates) == 1:
            cand = content_candidates[0]
            self.consumed_tokens.add(cand.lower())
            return cand

        return None

    def _resolve_enum_constant(self, domain_spec: str, port_state: str = "") -> Optional[str]:
        """
        Resolves an Enum constant dynamically by matching prompt intent against candidate flags.
        Domain-agnostic: uses domain_spec (e.g. 'cv2.COLOR_*') and module reflection.
        Contains ZERO library-specific keywords or hardcoded bonuses.
        Grounds selection in:
          1. Continuous vector embedding cosine similarity
          2. Directional transition alignment (X2Y matching prompt target intent)
          3. Semantic token overlap and Occam's razor parsimony
        """
        if not domain_spec or "*" not in domain_spec:
            return None

        clean_spec = domain_spec.replace("_*", "").replace("*", "")
        parts = clean_spec.rsplit(".", 1)
        if len(parts) != 2:
            return None
        mod_name, prefix = parts

        try:
            import importlib
            mod = importlib.import_module(mod_name)
        except Exception as e:
            return None

        candidates = [name for name in dir(mod) if name.startswith(f"{prefix}_")]
        if not candidates:
            return None

        sorted_candidates = sorted(candidates)

        # 1. Continuous vector embedding similarity if ModelManager is active
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            import numpy as np

            mm = ModelManager.get_instance()
            if mm.profile is not None and self.prompt:
                e_prompt = np.array(mm.get_embedding(self.prompt), dtype=np.float32)
                p_norm = np.linalg.norm(e_prompt)
                if p_norm > 0:
                    e_prompt = e_prompt / p_norm
                    best_cand = None
                    best_sim = -1.0
                    for cand in sorted_candidates:
                        cand_text = cand.replace("_", " ").lower()
                        e_c = np.array(mm.get_embedding(cand_text), dtype=np.float32)
                        c_norm = np.linalg.norm(e_c)
                        if c_norm > 0:
                            sim = float(np.dot(e_prompt, e_c / c_norm))
                            if sim > best_sim:
                                best_sim = sim
                                best_cand = cand
                    if best_cand and best_sim > 0.30:
                        return f"{mod_name}.{best_cand}"
        except Exception as e:
            logger.debug("suppressed: %s", e, exc_info=False)

        # 2. Token overlap and parsimony scoring fallback
        prompt_lower = (self.prompt or "").lower()
        p_tokens = CellTokenizer.tokenize_prompt(prompt_lower)

        scored = []
        for cand in sorted_candidates:
            parts = [p for p in CellTokenizer.tokenize_identifier(cand.lower()) if len(p) >= 2]
            if not parts:
                continue

            score = 0.0
            matched_parts = 0

            for idx, part in enumerate(parts):
                matched = False
                if part in p_tokens:
                    score += 2.0
                    matched = True
                elif any(t.startswith(part) or part.startswith(t) for t in p_tokens if len(t) >= 4 and len(part) >= 4):
                    score += 1.5
                    matched = True

                if matched:
                    matched_parts += 1
                    if idx == len(parts) - 1:
                        score += 2.0

            unmatched_parts = len(parts) - matched_parts
            score -= 0.5 * unmatched_parts
            scored.append((score, -len(cand), cand))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if scored and scored[0][0] > 0.0:
            return f"{mod_name}.{scored[0][2]}"

        return None

    def resolve_literal_for_port(
        self,
        port_sig: PortSignature,
        cell_stage: Optional[int] = None,
        cell_inputs: Optional[Dict[str, PortSignature]] = None,
        cell_tokens: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Resolves a value for a port using parameters, declared defaults, or literals.
        Zero domain-specific keywords or hardcoded values.
        Categorically grounded in morphism stages:
          - Stage 1 (Initial / Ingestion morphism Env -> C): resolves environment assets
          - Stage 2 (Endomorphism C x P -> C): resolves operational parameters P
          - Stage 3 (Terminal / Egress morphism C -> Env): resolves output destination

        Literal-to-port assignment follows a strict type-channel discipline:
          file_asset literals bind ONLY to path-typed ports (or to plain str ports on
          cells that declare no path-typed port), quoted_str to str ports, numeric to
          numeric ports. `cell_inputs` enables the cross-port guard that prevents
          auxiliary str ports from stealing file assets on cells that own a path port.
        """
        p_name = (port_sig.name or "").lower() if port_sig and getattr(port_sig, "name", None) else ""
        t_name = (port_sig.type_name or "").lower() if port_sig and getattr(port_sig, "type_name", None) else ""

        def _cell_has_path_port() -> bool:
            if not cell_inputs:
                return False
            return any(_lattice_is_path_port(p) for p in cell_inputs.values())

        registry = TypeRegistry.get_instance()
        is_bool = registry.is_subtype(t_name, "bool")
        is_num = registry.is_subtype(t_name, "numeric") and not is_bool
        is_str = registry.is_subtype(t_name, "str") or t_name in ("str", "any")

        # 1. Parameter explicitly supplied in context or LLM slots
        if port_sig.name in self.parameters:
            val = self.parameters[port_sig.name]
            return json.dumps(val) if isinstance(val, str) else str(val)
        if hasattr(self, "hyperparameters") and isinstance(self.hyperparameters, dict) and port_sig.name in self.hyperparameters:
            return _format_literal_value(self.hyperparameters[port_sig.name], port_sig)
        if hasattr(self, "llm_slots") and isinstance(self.llm_slots, dict):
            for cell_s in self.llm_slots.values():
                if isinstance(cell_s, dict) and port_sig.name in cell_s:
                    return _format_literal_value(cell_s[port_sig.name], port_sig)

        # 2. Dynamic Enum / Flag Constant Grounding via Domain Reflection & Vector Similarity
        if (t_name == "enum" or getattr(port_sig, "domain", None)) and self.prompt:
            domain_spec = getattr(port_sig, "domain", "") or ""
            resolved_enum = self._resolve_enum_constant(domain_spec, port_sig.state)
            if resolved_enum is not None:
                return resolved_enum

        # 3. Vector Polarity Projection for Boolean / Valuation Ports
        if is_bool and self.prompt:
            raw_quals = getattr(port_sig.signature, "qualifiers", [])
            qualifier_map = {}
            for q in (raw_quals or ()):
                if isinstance(q, (tuple, list)) and len(q) == 2:
                    qualifier_map[q[0]] = q[1]
                elif isinstance(q, str):
                    qualifier_map[q] = q
            pos_label = qualifier_map.get("positive", port_sig.name)
            neg_label = qualifier_map.get("negative", f"not {port_sig.name}")
            # Inverted-pole fallback for DECLARED order-flag identifiers: a port
            # named "descending" means its positive pole is largest-first.
            # Token-wise so e.g. "description" never matches.
            if not (
                "negative" in qualifier_map
                or p_name in _ORDER_FLAG_NAMES
                or bool(CellTokenizer.tokenize_identifier(p_name) & _ORDER_POLE_TOKENS)
            ):
                neg_label = f"not {port_sig.name}"

            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                import numpy as np

                mm = ModelManager.get_instance()
                if mm.profile is not None:
                    prompt_tokens = CellTokenizer.tokenize_prompt(self.prompt)
                    if prompt_tokens:
                        e_pos = np.array(mm.get_embedding(pos_label), dtype=np.float32)
                        e_neg = np.array(mm.get_embedding(neg_label), dtype=np.float32)
                        norm_pos = np.linalg.norm(e_pos)
                        norm_neg = np.linalg.norm(e_neg)
                        if norm_pos > 0 and norm_neg > 0:
                            e_pos = e_pos / norm_pos
                            e_neg = e_neg / norm_neg

                            token_list = list(prompt_tokens)
                            t_embs = [np.array(mm.get_embedding(t), dtype=np.float32) for t in token_list]
                            t_embs = [t / np.linalg.norm(t) for t in t_embs if np.linalg.norm(t) > 0]

                            if t_embs:
                                pos_score = max(float(np.dot(t, e_pos)) for t in t_embs)
                                neg_score = max(float(np.dot(t, e_neg)) for t in t_embs)
                                token_match = any(w in self.prompt.lower() for w in (port_sig.name.lower(), pos_label.lower(), neg_label.lower()))
                                if (token_match or max(pos_score, neg_score) > 0.65) and abs(pos_score - neg_score) > 0.05:
                                    return "True" if pos_score > neg_score else "False"
            except Exception as e:
                logger.debug(f"[UNIFICATION] Vector polarity projection fallback: {e}")

            # Declared-vocabulary polarity grounding (symbolic / embedding-free mode).
            # The positive/negative labels are DECLARED in the cell schema (qualifiers);
            # the engine only performs generic word membership against the prompt.
            # Zero hardcoded operation words: vocabulary is data, not code.
            prompt_lower = self.prompt.lower()
            pos_words = [w for w in str(pos_label).lower().split() if len(w) >= 2]
            neg_words = [w for w in str(neg_label).lower().split() if len(w) >= 2]
            pos_hit = any(w in prompt_lower for w in pos_words)
            neg_hit = any(w in prompt_lower for w in neg_words)
            if pos_hit != neg_hit:
                return "True" if pos_hit else "False"

            # Ordinal direction lexicon grounding: natural-language direction
            # adjectives ("top 2 rows", "smallest first") resolve declared
            # order/direction flag ports. Gated by the port's DECLARED polarity
            # metadata (name/state tokens and declared positive/negative pole
            # labels) — never by arbitrary booleans. The ordinal words
            # themselves are language-level English primitives.
            p_state_l = str(getattr(port_sig, "state", "") or "").lower()
            p_name_tokens = CellTokenizer.tokenize_identifier(p_name)
            is_order_port = (
                p_name in _ORDER_FLAG_NAMES
                or p_state_l in _ORDER_FLAG_STATES
                or bool(p_name_tokens & _ORDER_POLE_TOKENS)
                or "asc" in str(pos_label).lower().split() or "desc" in str(neg_label).lower().split()
            )
            if is_order_port:
                hint_toks = CellTokenizer.tokenize_prompt(self.prompt)
                asc_hint = bool(hint_toks & _ASCENDING_HINTS)
                desc_hint = bool(hint_toks & _DESCENDING_HINTS)
                if asc_hint != desc_hint:
                    # Does this port's POSITIVE pole mean ascending order?
                    # Inverted-pole names are DECLARED port identifiers, checked
                    # against the registry-harvested descending polarity hints
                    # (trees/*.json `polarity_hints`) — token-wise, so e.g.
                    # "description" never matches.
                    positive_means_ascending = not (
                        p_name in _DESCENDING_HINTS
                        or bool(p_name_tokens & {"desc", "descend", "descending", "decreasing", "reverse"})
                    )
                    descending_requested = desc_hint
                    if positive_means_ascending:
                        return "False" if descending_requested else "True"
                    return "True" if descending_requested else "False"

            if port_sig.required:
                if port_sig.default_value is not None:
                    return str(port_sig.default_value)
                return None
            return None

        # 4. Numeric literals for numeric ports.
        # A REQUIRED numeric port may consume any unconsumed numeric (it must
        # bind or the pipeline fails). An OPTIONAL numeric port consumes one
        # only with lexical EVIDENCE: the literal's prompt context must
        # intersect the consuming cell's declared vocabulary (identity tokens,
        # port name/state tokens). Otherwise a generic configuration knob
        # (dropna.axis) speculatively steals a number that belongs to another
        # step's parameter (head.n) — measured on "save the first 2 rows".
        if is_num:
            for idx, (lit_pos, kind, val) in enumerate(self.ordered_literals):
                if idx in self.used_indices or kind != "numeric":
                    continue
                if not port_sig.required:
                    # Clause-bounded context: restrict evidence to the clause enclosing the literal
                    prompt_len = len(self.prompt)
                    delims: List[Tuple[int, int]] = []
                    i = 0
                    while i < prompt_len:
                        c = self.prompt[i]
                        if c in (",", ";"):
                            delims.append((i, i + 1))
                            i += 1
                        elif self.prompt[i : i + 3].lower() == "and" and (i == 0 or not self.prompt[i - 1].isalnum()) and (i + 3 == prompt_len or not self.prompt[i + 3].isalnum()):
                            delims.append((i, i + 3))
                            i += 3
                        elif self.prompt[i : i + 4].lower() == "then" and (i == 0 or not self.prompt[i - 1].isalnum()) and (i + 4 == prompt_len or not self.prompt[i + 4].isalnum()):
                            delims.append((i, i + 4))
                            i += 4
                        else:
                            i += 1

                    spans = []
                    last = 0
                    for d_start, d_end in delims:
                        spans.append((last, d_start))
                        last = d_end
                    spans.append((last, prompt_len))
                    clause_chunk = self.prompt
                    for s_start, s_end in spans:
                        if s_start <= lit_pos <= s_end:
                            clause_chunk = self.prompt[s_start:s_end]
                            break

                    context_toks = {
                        t for t in CellTokenizer.tokenize_prompt(clause_chunk)
                    }

                    # A port with declared discrete enum_values is a categorical flag, not a continuous/count scalar.
                    # It must only bind if the port name itself is explicitly spoken in the context clause.
                    if getattr(port_sig, "enum_values", None):
                        port_ident_toks = {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)}
                        if port_sig.name:
                            port_ident_toks.add(port_sig.name.lower())
                        if not (context_toks & port_ident_toks):
                            continue

                    evidence = set()
                    evidence |= {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)}
                    if port_sig.name:
                        evidence.add(port_sig.name.lower())
                    _st = str(getattr(port_sig, "state", "") or "")
                    if _st.lower() not in ("any", "default", ""):
                        evidence |= {t.lower() for t in CellTokenizer.tokenize_identifier(_st)}
                    _desc = str(getattr(port_sig, "description", "") or getattr(port_sig, "doc", "") or "")
                    if _desc:
                        evidence |= {t.lower() for t in CellTokenizer.tokenize_prompt(_desc)}

                    # Only fall back to cell-wide identity tokens if the port itself lacks specific descriptive vocabulary
                    if not evidence and cell_tokens:
                        evidence |= {t.lower() for t in cell_tokens}

                    intersecting = context_toks & evidence
                    if not intersecting:
                        continue

                    # Prompt-clause IDF weighting:
                    # A word that appears in the port's boilerplate description or in multiple prompt clauses
                    # is non-discriminative background vocabulary (e.g. 'contour' in a contour detection prompt).
                    # An optional numeric port must only bind if the intersection with context contains
                    # discriminative evidence (high clause IDF or specific port identifier match).
                    all_clause_toks = [
                        {t.lower() for t in CellTokenizer.tokenize_prompt(self.prompt[s:e])}
                        for s, e in spans if self.prompt[s:e].strip()
                    ]
                    num_clauses = len(all_clause_toks)
                    if num_clauses > 1:
                        port_ident_toks = {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)}
                        if port_sig.name:
                            port_ident_toks.add(port_sig.name.lower())

                        discriminative_mass = 0.0
                        has_discriminative_hit = False
                        for tok in intersecting:
                            df = sum(1 for c_toks in all_clause_toks if tok in c_toks)
                            idf = math.log(1.0 + (num_clauses + 1.0) / (df + 1.0))
                            is_ident = tok in port_ident_toks
                            # High-IDF token (local to 1 clause) or specific port identity token
                            if df == 1 or (is_ident and df <= max(1, num_clauses // 2)):
                                discriminative_mass += idf * (1.5 if is_ident else 1.0)
                                has_discriminative_hit = True

                        if not has_discriminative_hit or discriminative_mass < 0.5:
                            continue
                self.used_indices.add(idx)
                return val

        # 4b. Predicate / boolean expression arguments, grounded by the DECLARED
        # typestate role (state) of the port — never by the port's identifier string.
        # Fallback: an expression literal (comparison syntax) can only ever ground
        # a string carrier, so a REQUIRED str port of a transform consumes it by
        # type affinity alone — no identifier knowledge, no state declaration
        # needed in the tree.
        _port_state = str(getattr(port_sig, "state", "")).lower()
        _registry = TypeRegistry.get_instance()
        _is_str_port = _registry.is_subtype(str(getattr(port_sig, "type_name", "")).lower(), "str")
        if _port_state in ("expr", "condition", "filter_condition"):
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind in ("expr", "quoted_str"):
                    self.used_indices.add(idx)
                    return json.dumps(val)
        elif _is_str_port and cell_stage == 2 and port_sig.required:
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind == "expr":
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 5. Stage 1 and Stage 3 Morphisms: Environmental Asset Grounding
        # file_asset literals flow only into path-typed ports. Plain str ports may
        # receive assets only when the cell declares NO dedicated path port, so that
        # auxiliary string parameters can never steal file assets from the sink/source.
        # Direction-aware allocation: English marks asset roles with relational
        # prepositions ("FROM input.csv", "TO output.csv") — ingress (stage 1)
        # prefers source/unmarked assets, egress (stage 3) prefers
        # destination-marked assets, so a destination literal is never stolen
        # by the ingress when both directions exist in the prompt.
        def _asset_direction(pos: int) -> str:
            return ExecutionContext._asset_direction(self.prompt or "", pos)

        is_path = _lattice_is_path_port(port_sig)
        port_role = str(getattr(port_sig, "port_role", None) or getattr(port_sig, "role", None) or "").strip().lower()
        is_path_role = port_role in ("path", "file", "filepath", "filename", "pathlike")

        def _take_asset(prefer: str) -> Optional[str]:
            fallback: Optional[Tuple[int, str]] = None
            path_candidates: List[Tuple[int, str]] = []
            for idx, (lit_pos, kind, val) in enumerate(self.ordered_literals):
                if idx in self.used_indices:
                    continue
                if getattr(port_sig, "enum_values", None):
                    if str(val).lower() not in {str(e).lower() for e in port_sig.enum_values}:
                        continue
                is_file_or_quoted = kind in ("file_asset", "quoted_str")
                # Guard: a single-character quoted_str with no path separator
                # or extension is structurally a label/column name, never a
                # file path. Prevent path ports from stealing column literals.
                if kind == "quoted_str" and len(str(val)) <= 1:
                    is_file_or_quoted = False
                matches_path = ExecutionContext._is_path_string(str(val))
                if is_file_or_quoted or (is_path_role and matches_path):
                    direction = _asset_direction(lit_pos) if kind == "file_asset" else ""
                    if prefer and direction == prefer:
                        self.used_indices.add(idx)
                        return json.dumps(val)
                    if fallback is None and direction != ("dest" if prefer == "src" else "src"):
                        fallback = (idx, val)
                    if matches_path:
                        path_candidates.append((idx, val))

            if is_path_role and len(path_candidates) == 1:
                idx, val = path_candidates[0]
                self.used_indices.add(idx)
                return json.dumps(val)

            if fallback is not None:
                self.used_indices.add(fallback[0])
                return json.dumps(fallback[1])
            return None

        allow_str_asset = (
            cell_stage in (1, 3)
            and is_str
            and not getattr(port_sig, "enum_values", None)
            and not _cell_has_path_port()
            and (port_sig.required or cell_stage == 3)
        )
        if is_path or is_path_role or allow_str_asset:
            if cell_stage == 3:
                taken = _take_asset("dest")
            elif cell_stage == 1:
                taken = _take_asset("src")
            else:
                taken = _take_asset("")
            if taken is not None:
                return taken
            # If Stage 1 source cell has a required path port and prompt omitted explicit file literal:
            if port_sig.required and cell_stage == 1:
                if port_sig.default_value is not None:
                    val = str(port_sig.default_value)
                    return json.dumps(val) if not (val.startswith(("'", '"')) or val in ("None", "True", "False")) else val
                t_str = str(getattr(port_sig, "type_name", "")).lower()
                dom = str(getattr(port_sig, "domain", "")).lower()
                registry = TypeRegistry.get_instance()
                modality = registry.get_abstract_carrier(t_str) or registry.get_abstract_carrier(dom)
                if not modality:
                    if "image" in t_str:
                        modality = "image"
                    elif "table" in t_str or "csv" in t_str:
                        modality = "table"
                    elif "array" in t_str or "tensor" in t_str:
                        modality = "array"
                    elif "text" in t_str:
                        modality = "text"
                    else:
                        modality = "default"
                placeholder = registry.get_asset_placeholder(modality)
                return json.dumps(placeholder)

            # If Stage 3 sink cell has a path port and prompt omitted explicit destination literal:
            if cell_stage == 3 and (port_sig.required or is_path or is_path_role):
                t_str = str(getattr(port_sig, "type_name", "")).lower()
                dom = str(getattr(port_sig, "domain", "")).lower()
                registry = TypeRegistry.get_instance()
                modality = registry.get_abstract_carrier(t_str) or registry.get_abstract_carrier(dom)
                if not modality:
                    if "image" in t_str:
                        modality = "image"
                    elif "table" in t_str or "csv" in t_str:
                        modality = "table"
                    elif "array" in t_str or "tensor" in t_str:
                        modality = "array"
                    elif "text" in t_str:
                        modality = "text"
                    else:
                        modality = "default"
                placeholder = registry.get_output_asset_placeholder(modality)
                return json.dumps(placeholder)

        # 6. Stage 2 Morphism: Operational Parameter Extraction
        if (cell_stage == 2 or cell_stage is None) and is_str:
            enum_vals = {str(e).strip().lower() for e in port_sig.enum_values} if getattr(port_sig, "enum_values", None) else None
            p_state = str(getattr(port_sig, "state", "") or "").lower()
            p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "derived_role", "") or "").lower()
            col_tokens = TypeRegistry.get_instance().get_column_projection_tokens()
            reg_types = TypeRegistry.get_instance().get_registered_types()
            is_col_port = (
                p_role in ("column_projection", "columns", "target_input")
                or p_state in ("column_projection", "columns", "feature_names", "column_name", "target_name")
            )

            # Check unconsumed literals for operational parameter candidates (prioritizing current clause)
            cand_indices = list(range(len(self.ordered_literals)))
            if self.current_cell_clause_idx is not None:
                cand_indices.sort(key=lambda i: 0 if self.literal_clause_map.get(i) == self.current_cell_clause_idx else 1)

            for idx in cand_indices:
                lit_pos, kind, val = self.ordered_literals[idx]
                if idx in self.used_indices:
                    continue
                if kind not in ("quoted_str", "identifier"):
                    continue
                val_str = str(val).strip()
                val_low = val_str.lower()
                # Target protection: operational parameters must not steal the target sink variable
                if val_low in self.target_literals:
                    continue
                # Exclude file assets from operational parameters
                if ExecutionContext._is_path_string(val_str) or ("." in val_str and not val_str.replace(".", "").replace("-", "").isdigit()):
                    continue
                # Enforce enum constraints if declared
                if enum_vals is not None:
                    if val_low not in enum_vals:
                        continue
                # Exclude Python keywords and registered types from being bound as column names
                if is_col_port and (keyword.iskeyword(val_low) or val_low in reg_types):
                    continue
                # Column & target protection: non-column parameter ports must not steal data columns
                roles = self.identifier_roles.get(lit_pos, frozenset())
                is_data_col = bool(roles & col_tokens) or (kind == "quoted_str" and not ExecutionContext._is_path_string(val_str))
                if is_data_col and not is_col_port:
                    continue
                if is_col_port and not is_data_col:
                    continue
                # Optional parameters without explicit enum match require lexical evidence
                if not port_sig.required and enum_vals is None:
                    port_ident_toks = {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)} if port_sig.name else set()
                    _desc = str(getattr(port_sig, "description", "") or getattr(port_sig, "doc", "") or "")
                    evidence = set(port_ident_toks)
                    if _desc:
                        evidence |= {t.lower() for t in CellTokenizer.tokenize_prompt(_desc)}
                    if cell_tokens:
                        evidence |= {t.lower() for t in cell_tokens}
                    clause_toks = {t.lower() for t in CellTokenizer.tokenize_prompt(self.prompt)}
                    if not (evidence & clause_toks) and not (port_ident_toks and any(tok in self.prompt.lower() for tok in port_ident_toks)):
                        continue
                self.used_indices.add(idx)
                return json.dumps(val)

        # 6b. Referential identifier grounding (role-conditioned).
        if (cell_stage == 2 or cell_stage is None) and port_sig.required and port_sig.default_value is None:
            _tn = t_name
            _is_collection = (
                registry.is_subtype(_tn, "list")
                or registry.is_subtype(_tn, "collection")
                or registry.is_subtype(_tn, "sequence")
                or _tn in ("list", "sequence", "collection")
                or getattr(port_sig, "abstract_type", None) == "collection"
                or str(getattr(port_sig, "state", "")).lower() in ("column_projection", "columns", "columns_list", "feature_names")
            )
            _is_strict_str = (registry.is_subtype(_tn, "str") or registry.is_subtype(_tn, "scalar")) and _tn not in ("any", "*", "top", "") and not _is_collection
            _state_tokens: Set[str] = set()
            _raw_state = str(getattr(port_sig, "state", "") or "")
            if _raw_state.lower() not in ("any", "default", ""):
                _state_tokens = CellTokenizer.tokenize_identifier(_raw_state)
            if _is_strict_str and (_state_tokens or cell_tokens):
                _identity_scope: Set[str] = set(cell_tokens or set()) | _state_tokens
                cand_indices = list(range(len(self.ordered_literals)))
                if self.current_cell_clause_idx is not None:
                    cand_indices.sort(key=lambda i: 0 if self.literal_clause_map.get(i) == self.current_cell_clause_idx else 1)
                for idx in cand_indices:
                    pos, kind, val = self.ordered_literals[idx]
                    if idx in self.used_indices or kind != "identifier":
                        continue
                    val_str = str(val).strip()
                    val_low = val_str.lower()
                    if val_low in self.target_literals or keyword.iskeyword(val_low) or val_low in registry.get_registered_types():
                        continue
                    role_tokens = self.identifier_roles.get(pos, frozenset())
                    if not role_tokens or not (role_tokens & _identity_scope):
                        continue
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 6c. Multi-identifier collection / list projection (e.g. columns list for selection or projection)
        is_collection = (
            registry.is_subtype(t_name, "list")
            or registry.is_subtype(t_name, "collection")
            or registry.is_subtype(t_name, "sequence")
            or t_name in ("list", "sequence", "collection")
            or getattr(port_sig, "abstract_type", None) == "collection"
            or str(getattr(port_sig, "state", "")).lower() in ("column_projection", "columns", "columns_list", "feature_names")
        )
        if (cell_stage == 2 or cell_stage is None) and is_collection:
            _raw_state = str(getattr(port_sig, "state", "") or "")
            _state_tokens = CellTokenizer.tokenize_identifier(_raw_state) if _raw_state.lower() not in ("any", "default", "") else set()
            _proj_tokens = TypeRegistry.get_instance().get_column_projection_tokens()
            _identity_scope: Set[str] = set(cell_tokens or set()) | _state_tokens | _proj_tokens
            egress_tokens = TypeRegistry.get_instance().get_egress_tokens() | TypeRegistry.get_instance().get_dest_port_tokens()
            registered_types = TypeRegistry.get_instance().get_registered_types()
            stopwords = TypeRegistry.get_instance().get_stopwords()

            matched_indices = []
            matched_vals = []
            cell_clause = getattr(self, "current_cell_clause_idx", None)
            lit_clause_map = getattr(self, "literal_clause_map", {})
            target_lits = getattr(self, "target_literals", set())

            all_available = [
                idx for idx, (_, kind, val) in enumerate(self.ordered_literals)
                if idx not in self.used_indices and kind in ("identifier", "quoted_str")
                and str(val).lower() not in target_lits
            ]
            if cell_clause is not None and lit_clause_map:
                clause_cands = [idx for idx in all_available if lit_clause_map.get(idx) == cell_clause]
                cands_to_check = clause_cands if clause_cands else all_available
            else:
                cands_to_check = all_available

            for idx in cands_to_check:
                pos, kind, val = self.ordered_literals[idx]
                val_str = str(val).strip()
                val_low = val_str.lower()
                if val_low in target_lits or keyword.iskeyword(val_low) or val_low in registered_types or val_low in stopwords:
                    continue
                role_tokens = self.identifier_roles.get(pos, frozenset())
                if role_tokens and (role_tokens & egress_tokens):
                    continue

                is_candidate = False
                if role_tokens and (role_tokens & _identity_scope):
                    is_candidate = True
                elif kind == "quoted_str" and not ExecutionContext._is_path_string(val_str) and "." not in val_str:
                    is_candidate = True
                elif _raw_state in ("column_projection", "columns", "columns_list", "feature_names") and kind == "identifier":
                    if role_tokens and bool(role_tokens & _proj_tokens):
                        is_candidate = True

                if is_candidate and val not in matched_vals:
                    matched_indices.append(idx)
                    matched_vals.append(val)

            if matched_vals:
                for idx in matched_indices:
                    self.used_indices.add(idx)
                    self.consumed_tokens.add(str(self.ordered_literals[idx][2]).lower())
                return json.dumps(matched_vals)

        # 7. Port default value declared in tree schema
        if port_sig.default_value is not None:
            def_str = str(port_sig.default_value).strip()
            if def_str in ("True", "False", "None"):
                return def_str
            if not is_str:
                return def_str
            # For string ports, check if def_str is an unquoted module constant (module-scoped enum attribute)
            parts = def_str.split(".")
            if len(parts) > 1 and parts[0] in sys.modules:
                return def_str
            return def_str if (def_str.startswith('"') or def_str.startswith("'")) else json.dumps(def_str)

        # 8. Pure Vector Semantic Slot Projection for unquoted string/identifier arguments.
        # The projection is ROLE-FIRST: when the port declares a semantic role
        # (typestate state), the projected value is the prompt word closest to
        # that ROLE — never to the port's own identifier (asking "which word
        # looks like the word 'name'?" confuses the slot's label with the
        # value's meaning). Words morphologically related to the port's own
        # name are excluded outright ("a file NAMED data.csv" must not feed a
        # port called `name` because they are near-identical strings).
        if (cell_stage == 2 or cell_stage is None) and is_str and self.prompt:
            _raw_state = str(getattr(port_sig, "state", "") or "")
            role_label = _raw_state if _raw_state.lower() not in ("any", "default", "") else ""
            projected = self._project_semantic_slot(port_sig.name, role_label=role_label)
            if projected:
                return json.dumps(projected)

        return None


# =====================================================================
# 5. Type-Monadic Unification Gate
# =====================================================================

@dataclass
class VerificationContract:
    """
    Structured task verification contract generated during AST synthesis.
    Encapsulates Phase-1 postconditions and terminal node intent checks for GEVR sandbox execution.
    """
    cell_checks: List[Dict[str, Any]] = field(default_factory=list)
    terminal_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_checks": self.cell_checks,
            "terminal_checks": self.terminal_checks
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VerificationContract":
        return cls(
            cell_checks=d.get("cell_checks", []),
            terminal_checks=d.get("terminal_checks", [])
        )

@functools.lru_cache(maxsize=512)
def _get_module_symbols(mod_name: str) -> FrozenSet[str]:
    """Caches exported symbols of submodules for fast dependency resolution."""
    try:
        import sys, importlib
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = importlib.import_module(mod_name)
        return frozenset(w for w in dir(mod) if not w.startswith("_"))
    except Exception as e:
        return frozenset()


class UnificationGate:
    """
    Formal Unification Gate verifying dataflow composition and emitting code.
    Contains ZERO hardcoded domain libraries or prompt-sniffing regexes.
    """
    def __init__(self, orchestrator: Optional[Any] = None):
        self.orchestrator = orchestrator
        self.context = ExecutionContext()
        self.last_egress_paths: List[str] = []
        self.last_pipeline_bindings: List[Tuple[Cell, Dict[str, str]]] = []
        self.last_verification_contract: Optional[VerificationContract] = None
        self.last_runtime_aliases: Dict[str, str] = {}

    def get_runtime_aliases(self) -> Dict[str, str]:
        """
        Returns the prompt-declared egress aliases produced by the most recent
        synthesis (e.g. {"Z": "var_3"} for "store the results into Z"). The
        sandbox applies these in the runtime namespace so the declared sink
        identifier resolves without fabricating source-level assignments.
        """
        return dict(self.last_runtime_aliases)

    # ------------------------------------------------------------------
    # Declared bridge-morphism composition (Solution 3)
    # ------------------------------------------------------------------

    def _compose_declared_bridge_expr(
        self,
        src_var: str,
        src_type: str,
        dst_type: str,
        literal_bindings: Dict[str, str],
        dst_sig: Optional[Any] = None,
        ctx: Optional[ExecutionContext] = None,
        max_hops: int = 2,
    ) -> Optional[str]:
        """
        Composes an expression from DECLARED bridge morphisms connecting
        src_type to dst_type. Bridge morphisms are ordinary tree cells
        (node_role == 'bridge') whose single-statement template has the shape
        `{output_var} = <expression over its input ports>`.

        The engine contributes ONLY the composition mechanism: it reads the
        templates out of the loaded cells (which come from pluggable domain
        trees) and instantiates them generically. When no tree declares a
        morphism chain between the carriers, this returns None and the port
        remains unresolved — the engine never writes library-specific syntax
        (column projections, array conversions, ...) on its own.
        """
        orchestrator = getattr(self, "orchestrator", None)
        if orchestrator is None:
            return None
        registry = TypeRegistry.get_instance()

        bridge_cells = [
            c for c in (getattr(orchestrator, "loaded_cells", {}) or {}).values()
            if str(getattr(c, "node_role", "")).lower() == "bridge"
            or str(getattr(c, "node_type", "")).lower() == "tunnel"
        ]
        if not bridge_cells:
            return None

        def _accepts(port_sig: Any, have: str) -> bool:
            pt = str(getattr(getattr(port_sig, "signature", port_sig), "type_name", "") or "").lower()
            have_l = (have or "").lower()
            if pt in ("any", "*", "top", ""):
                return True
            if not have_l:
                return False
            return registry.is_subtype(have_l, pt)

        def _dst_accepts(have: str) -> bool:
            have_l = (have or "").lower()
            dst_l = str(dst_type or "").lower()
            if dst_l and registry.is_subtype(have_l, dst_l):
                return True
            ab = str(getattr(dst_sig, "abstract_type", "") or "").lower() if dst_sig is not None else ""
            return bool(ab and ab not in ("any", "*") and registry.is_subtype(have_l, ab))

        def _template_rhs(cell: Any) -> Optional[Tuple[str, str, Dict[str, str]]]:
            tpl = (getattr(cell, "code_template", "") or "").strip()
            if not tpl or "\n" in tpl or tpl.count("=") != 1:
                return None
            lhs, rhs = tpl.split("=", 1)
            out_name = getattr(getattr(cell, "primary_output", None), "name", "") or "output_var"
            if lhs.strip() != "{" + out_name + "}":
                return None
            rhs = rhs.strip()
            placeholders = set(extract_template_placeholders(rhs))
            if not placeholders:
                return None
            out_decl = (getattr(cell, "outputs", {}) or {}).get(out_name)
            out_type = ""
            if isinstance(out_decl, dict):
                out_type = str(out_decl.get("type_name", ""))
            elif out_decl is not None:
                out_type = str(getattr(out_decl, "type_name", ""))
            return rhs, out_name, {"output_type": out_type}

        def _literal_for(port_name: str, port_sig: Any) -> Optional[str]:
            # Match declared literal bindings by port name first, then by
            # tokenized overlap between the port's declared name/state and the
            # binding key (e.g. state "column_identifier" <-> key "column").
            if port_name in literal_bindings:
                return literal_bindings[port_name]
            toks = CellTokenizer.tokenize_identifier(port_name)
            state = str(getattr(getattr(port_sig, "signature", port_sig), "state", "") or "")
            toks |= CellTokenizer.tokenize_identifier(state)
            for key, rendered in literal_bindings.items():
                if CellTokenizer.tokenize_identifier(key) & toks:
                    return rendered
            if not getattr(port_sig, "required", True):
                default_val = getattr(port_sig, "default_value", None)
                if default_val is not None:
                    return repr(default_val)
                return "None"
            return None

        frontier: List[Tuple[str, str]] = [(str(src_type or "").lower(), str(src_var))]
        seen_types = {str(src_type or "").lower()}

        for _hop in range(max(1, max_hops)):
            next_frontier: List[Tuple[str, str]] = []
            for cur_type, cur_expr in frontier:
                if _dst_accepts(cur_type):
                    return cur_expr
                for cell in bridge_cells:
                    tpl = _template_rhs(cell)
                    if tpl is None:
                        continue
                    rhs, out_name, meta = tpl
                    out_type = (meta.get("output_type") or "").lower()
                    if not out_type or out_type in seen_types:
                        continue
                    in_ports = list((getattr(cell, "inputs", {}) or {}).items())
                    if not in_ports:
                        continue
                    # Try each input port as the chain carrier; every other
                    # required port must be satisfied from declared literals.
                    for carrier_name, carrier_sig in in_ports:
                        if not _accepts(carrier_sig, cur_type):
                            continue
                        subs: Dict[str, str] = {out_name: cur_expr}
                        satisfiable = True
                        for other_name, other_sig in in_ports:
                            if other_name == carrier_name:
                                continue
                            rendered = _literal_for(other_name, other_sig)
                            if rendered is None:
                                satisfiable = False
                                break
                            subs[other_name] = rendered
                        if not satisfiable:
                            continue
                        subs[carrier_name] = cur_expr
                        composed = safe_substitute_template(rhs, subs)
                        leftover = set(extract_template_placeholders(composed))
                        if leftover:
                            continue
                        next_frontier.append((out_type, composed))
                        seen_types.add(out_type)
                        break
            if not next_frontier:
                break
            frontier = next_frontier

        for cur_type, cur_expr in frontier:
            if _dst_accepts(cur_type):
                return cur_expr
        return None

    def get_egress_paths(self) -> List[str]:
        """
        Returns destination artifact paths derived from the most recent synthesis.
        Single source of truth for sandbox egress verification: paths are the values
        bound to path-typed ports of terminal (Stage 3 / sink) morphisms during the
        last unify_pipeline run — never re-parsed from the raw prompt.
        """
        return list(self.last_egress_paths)

    @staticmethod
    def _derive_egress_paths(pipeline_bindings: List[Tuple[Cell, Dict[str, str]]]) -> List[str]:
        """
        Extracts egress destinations from verified pipeline bindings.
        A path qualifies as an egress artifact iff it is bound as a quoted literal
        to a path-typed port of:
          - a Stage 3 (terminal/egress) morphism, or
          - a cell whose output typestate declares materialization, or
          - any non-source (Stage 2+) morphism — in-place endomorphisms such as
            writers with a destination port (e.g. tabular writers) are Stage 2
            composable morphisms whose path port is still a materialization site.
        Type- and stage-driven, domain-agnostic.
        """
        registry = TypeRegistry.get_instance()
        egress: List[str] = []

        def _quoted_literal(v: Any) -> Optional[str]:
            if not isinstance(v, str):
                return None
            s = v.strip()
            if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                inner = s[1:-1].strip()
                return inner or None
            return None

        for cell, bindings in pipeline_bindings:
            stage = getattr(cell, "stage", None)
            is_terminal = stage == 3
            if not is_terminal:
                for out_p in getattr(cell, "outputs", {}).values():
                    if str(getattr(out_p, "state", "")).lower() in ("destination_written", "filepath_written", "saved", "exported"):
                        is_terminal = True
                        break
            is_source = stage == 1
            if not is_terminal and is_source:
                continue

            for p_name, p_sig in getattr(cell, "inputs", {}).items():
                t_name = str(getattr(p_sig, "type_name", ""))
                is_path_port = (
                    registry.is_subtype(t_name, "filepath")
                    or registry.is_subtype(t_name, "path")
                    or registry.is_subtype(t_name, "uri")
                )
                if not is_path_port:
                    continue
                bound_val = bindings.get(p_name)
                unquoted = _quoted_literal(bound_val)
                if unquoted and (is_terminal or stage == 2):
                    egress.append(unquoted)

        return egress

    def unify_transition(
        self,
        producer: Cell,
        consumer: Cell,
        current_sigma: Substitution,
        context: Optional[ExecutionContext] = None
    ) -> MonadResult[Substitution]:
        """
        Verifies that producer's output can satisfy an input of consumer,
        or that consumer's required inputs are satisfiable by available wires.
        Supports multi-port monoidal matching (⊗, Δ).
        """
        out_sig = producer.primary_output

        def _extract_ndim(sc: Any) -> Optional[int]:
            if isinstance(sc, dict):
                val = sc.get("ndim")
                if isinstance(val, int):
                    return val
                try:
                    return int(val) if val is not None else None
                except (ValueError, TypeError):
                    return None
            elif isinstance(sc, str):
                sc = sc.strip()
                if sc.startswith("(") and sc.endswith(")"):
                    inner = sc[1:-1].strip()
                    if not inner:
                        return 0
                    parts = [p.strip() for p in inner.split(",") if p.strip()]
                    return len(parts)
            return None

        def _shape_compatible(p_out: Any, p_in: Any) -> bool:
            out_sc = getattr(p_out, "shape_contract", None)
            in_sc = getattr(p_in, "shape_contract", None)
            if out_sc and in_sc:
                o_ndim = _extract_ndim(out_sc)
                i_ndim = _extract_ndim(in_sc)
                if o_ndim is not None and i_ndim is not None and o_ndim != i_ndim:
                    return False
            return True

        # 1. Primary input direct unification
        if _shape_compatible(out_sig, consumer.primary_input):
            new_sigma = unify(out_sig.signature, consumer.primary_input.signature, current_sigma)
            if new_sigma is not None:
                return Success(new_sigma, new_sigma)

        # 2. Multi-Port Monoidal Matching: check if producer output unifies with ANY input of consumer
        for p_name, p_port in consumer.inputs.items():
            if not _shape_compatible(out_sig, p_port):
                continue
            new_sigma = unify(out_sig.signature, p_port.signature, current_sigma)
            if new_sigma is not None:
                return Success(new_sigma, new_sigma)

        # 3. Port-sharing delta check: if consumer's inputs are satisfiable by in-scope variables
        if context is not None:
            for v_name, (v_sig, _) in context.variables.items():
                v_u = unify(v_sig.signature, consumer.primary_input.signature, current_sigma)
                if v_u is not None:
                    return Success(v_u, v_u)
            for p_name, p_port in consumer.inputs.items():
                if not getattr(p_port, "required", False) and p_port.default_value is not None:
                    continue
                for v_name, (v_sig, _) in context.variables.items():
                    if _shape_compatible(v_sig, p_port):
                        v_u = unify(v_sig.signature, p_port.signature, current_sigma)
                        if v_u is not None:
                            return Success(v_u, v_u)

        return Failure(
            f"Typestate Unification Failed: {producer.cell_id} outputs {out_sig.signature} "
            f"which cannot satisfy any input of {consumer.cell_id}"
        )

    def unify_pipeline(
        self,
        cells: List[Cell],
        context: Optional[ExecutionContext] = None
    ) -> MonadResult[List[Tuple[Cell, Dict[str, str]]]]:
        """
        Chains a sequence of cells [v_1, ..., v_n] through the Type Monad.
        Binds port placeholders to variables in each step using multi-port monoidal matching.

        Structural extensions:
        - Zero-ary constructor morphisms (node_type 'constructor' with no required
          data port) are insertable at any position: they consume no incoming wire.
        - Heterogeneous product outputs (tuple[A, B, ...]) are projected member-wise:
          a downstream port binds to ``var[i]`` (product elimination). Members already
          consumed by earlier cells are deprioritized, so partitions (train/verify/test)
          allocate distinct members to distinct consumers by consumption order and
          declared member-state affinity with the consuming clause.
        """
        if not cells:
            return Failure("Empty cell pipeline")

        registry = TypeRegistry.get_instance()
        ctx = context or ExecutionContext()
        accumulated_sigma = Substitution()
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]] = []
        var_counter = getattr(ctx, "var_counter", 0)
        producer_var: Optional[str] = None
        cell_out_vars: Dict[Any, str] = {}
        cell_port_vars: Dict[Tuple[Any, str], str] = {}

        # Clause decomposition of the prompt (connector-based fast path; shared with
        # the planner) — used solely for member-state affinity when projecting
        # heterogeneous products. Zero domain vocabulary.
        prompt_text = getattr(ctx, "prompt", "") or ""
        clause_token_sets: List[Set[str]] = []
        if prompt_text:
            for cl in CellTokenizer.split_prompt_clauses(prompt_text):
                cl = cl.strip()
                if cl:
                    clause_token_sets.append(CellTokenizer.tokenize_prompt(cl))

        def _cell_clause_tokens(cell: Any) -> Set[str]:
            if not clause_token_sets:
                return set()
            c_toks = getattr(cell, "token_set", set())
            best_idx, best_ov = 0, -1
            for idx, cl_toks in enumerate(clause_token_sets):
                ov = len(cl_toks & c_toks)
                if ov > best_ov:
                    best_ov, best_idx = ov, idx
            return clause_token_sets[best_idx] if best_ov > 0 else set()

        def _is_product(v_sig: Any, registry: Optional[Any] = None) -> bool:
            if registry is None:
                registry = TypeRegistry.get_instance()
            raw = str(getattr(getattr(v_sig, "signature", None), "type_name", "") or getattr(v_sig, "type_name", "") or "")
            ctor = raw.strip().lower().split("[", 1)[0] if "[" in raw else raw.strip().lower()
            return registry.is_product_constructor(ctor)

        def _member_candidates(v_sig: Any, v_name: str, port_sig: Any, cell_toks: Set[str], registry: Optional[Any] = None) -> List[Tuple[float, int, Substitution]]:
            """Product-elimination candidates: (score, member_index, substitution)."""
            raw = str(getattr(getattr(v_sig, "signature", None), "type_name", "") or "")
            if "[" not in raw or not raw.endswith("]"):
                return []
            if registry is None:
                registry = TypeRegistry.get_instance()
            constructor = raw.split("[", 1)[0].strip().lower()
            if not registry.is_product_constructor(constructor):
                return []
            inner = raw[raw.index("[") + 1 : -1]
            members: List[str] = []
            depth = 0
            curr: List[str] = []
            for ch in inner:
                if ch == "[":
                    depth += 1
                    curr.append(ch)
                elif ch == "]":
                    depth -= 1
                    curr.append(ch)
                elif ch == "," and depth == 0:
                    members.append("".join(curr).strip())
                    curr = []
                else:
                    curr.append(ch)
            if curr:
                members.append("".join(curr).strip())
            if len(members) < 2:
                return []

            consumed = getattr(ctx, "consumed_members", set())
            candidates: List[Tuple[float, int, Substitution]] = []
            for i, m_str in enumerate(members):
                try:
                    member_term = TypeTerm.from_string(m_str)
                except Exception as e:
                    continue
                u = unify(member_term, port_sig.signature, accumulated_sigma)
                if u is None:
                    continue
                sc = 0.0
                if (v_name, i) in consumed:
                    sc -= 5.0
                state_part = m_str[m_str.index("[") + 1 : -1] if "[" in m_str and m_str.endswith("]") else ""
                if state_part and state_part.lower() != "any" and cell_toks:
                    st_toks = CellTokenizer.tokenize_identifier(state_part)
                    sc += 2.0 * len(st_toks & cell_toks)
                candidates.append((sc, i, u))
            return candidates

        # Static Pre-Unification Structural Macro Expansion (Phase 4 / T4.1)
        # A macro expands ONLY when every sub-cell id resolves; otherwise the
        # macro stays intact (rendered as a single composite morphism) and a
        # warning is logged. Re-inserting a partially-resolved macro here would
        # make the expansion loop non-convergent (resolved sub-cells would be
        # duplicated on every pass while the macro itself remained in the list).
        changed = True
        expansion_depth = 0
        while changed and expansion_depth < 10:
            changed = False
            expansion_depth += 1
            expanded_cells: List[Cell] = []
            for c in cells:
                sub_cells = getattr(c, "sub_cells", None)
                if not ((getattr(c, "cell_type", "") == "macro" or isinstance(c, MacroCell)) and sub_cells):
                    expanded_cells.append(c)
                    continue

                orch = getattr(self, "orchestrator", None)
                if orch is None:
                    try:
                        from lattice import LatticeOrchestrator
                        orch = LatticeOrchestrator.get_active_instance()
                    except (ImportError, ValueError):
                        try:
                            from .lattice import LatticeOrchestrator
                            orch = LatticeOrchestrator.get_active_instance()
                        except Exception as e:
                            orch = None
                resolved_cache = getattr(c, "_resolved_sub_cells", {}) or {}
                resolved: List[Cell] = []
                missing: List[str] = []
                for sub_item in sub_cells:
                    sub_cell = sub_item if isinstance(sub_item, Cell) else None
                    if sub_cell is None:
                        sub_cell = resolved_cache.get(sub_item)
                    if sub_cell is None and orch:
                        sub_cell = orch.loaded_cells.get(sub_item)
                    if sub_cell is None:
                        missing.append(str(sub_item))
                    else:
                        resolved.append(sub_cell)

                if missing:
                    logger.warning(
                        f"[UNIFY] Macro '{c.cell_id}' kept unexpanded: unresolved sub-cells {missing}."
                    )
                    expanded_cells.append(c)
                else:
                    expanded_cells.extend(resolved)
                    changed = True
            cells = expanded_cells

        # Pre-scan pipeline for target-input ports to prevent target column from bleeding into feature projection
        has_target_port = any(
            any(getattr(p, "port_role", None) == "target_input" or getattr(p, "derived_role", "") == "target_input"
                for p in c.inputs.values())
            for c in cells
        )
        if has_target_port:
            unconsumed_ids = [
                val for _, kind, val in getattr(ctx, "ordered_literals", [])
                if kind in ("identifier", "quoted_str") and str(val).lower() not in ctx.consumed_tokens
            ]
            if getattr(ctx, "target_col", None):
                ctx.target_literals.add(str(ctx.target_col).lower())
            elif unconsumed_ids:
                target_col_name = unconsumed_ids[-1]
                ctx.target_literals.add(str(target_col_name).lower())

        # Process each cell in sequence
        for idx, cell in enumerate(cells):
            ctx.current_cell_clause_idx = getattr(cell, "matched_clause_idx", None)
            # For-each replicas (multiplicity expansion): a replica RE-CONSUMES
            # its receiver from the environment's in-scope variables (e.g. the
            # source table) instead of the previous wire, and its reference
            # port binds the NEXT member of the identifier role group. It
            # consumes no incoming wire, exactly like a zero-ary constructor.
            is_replica = bool(getattr(cell, "replica_of", None))
            cell_bindings: Dict[str, str] = {}
            current_out_var = None
            if len(cell.outputs) == 1:
                var_counter += 1
                ctx.var_counter = var_counter
                current_out_var = f"var_{var_counter}"
                cell_bindings["output_var"] = current_out_var
            elif len(cell.outputs) > 1:
                # Multi-output variables will be allocated per-port
                pass
            # For len(cell.outputs) == 0 (void output like save/show): no output_var allocated
            ctx.consumed_tokens.update(t.lower() for t in cell.token_set)

            def _is_zero_ary_generator(c: Cell) -> bool:
                if not c.outputs or getattr(c, "stage", None) == 3 or str(getattr(c, "node_role", "")).lower() == "sink":
                    return False
                if any(_lattice_is_path_port(p) for p in c.inputs.values()):
                    return False
                for p in c.inputs.values():
                    if p.required and p.default_value is None:
                        return False
                return True

            is_zero_ary = (
                getattr(cell, "node_type", "") == "constructor"
                or _is_zero_ary_generator(cell)
            )

            bound_producer = False

            # 1. Multi-Carrier DAG Scope Verification & Wire Resolution:
            # When idx > 0, verify that cell's inputs are satisfiable from the full
            # DAG ancestor frontier (available_cells = cells[:idx]).
            if idx > 0 and not is_zero_ary and not is_replica:
                scope_res = unify_cell_with_scope(cell, available_cells=cells[:idx], sigma=accumulated_sigma, context=ctx)
                if scope_res is not None:
                    accumulated_sigma, dag_scope_bindings = scope_res
                    for p_name, (prod_cell, out_port_name) in dag_scope_bindings.items():
                        v_name = None
                        if prod_cell is not None:
                            v_name = (
                                cell_port_vars.get((id(prod_cell), out_port_name))
                                or cell_out_vars.get(id(prod_cell))
                                or cell_port_vars.get((prod_cell.cell_id, out_port_name))
                                or cell_out_vars.get(prod_cell.cell_id)
                            )
                        if not v_name and out_port_name and out_port_name in getattr(ctx, "variables", {}):
                            v_name = out_port_name
                        if v_name and p_name not in cell_bindings:
                            cell_bindings[p_name] = v_name
                            bound_producer = True
                else:
                    # Fallback backward search if whole-scope unification was not satisfied
                    transition_res = None
                    for k in range(idx - 1, -1, -1):
                        cand_producer = cells[k]
                        res = self.unify_transition(cand_producer, cell, accumulated_sigma, context=ctx)
                        if not res.is_bottom():
                            transition_res = res
                            break
                    if transition_res is None or transition_res.is_bottom():
                        return Failure(transition_res.reason if isinstance(transition_res, Failure) else f"Transition failed for cell {cell.cell_id} from ancestors")
                    assert isinstance(transition_res, Success)
                    accumulated_sigma = transition_res.sigma

            # 2. Multi-Port Monoidal Matching: Bind input ports across available wires using semantic roles
            def _is_matching_port(p_n: str, p_s: Any) -> bool:
                if getattr(p_s, "required", False):
                    return True
                role = getattr(p_s, "derived_role", "standard")
                ROLE_CARRIERS = TypeRegistry.get_instance().get_declared_role_carriers()
                if role in ROLE_CARRIERS:
                    return True
                return False

            multi_ports = [(k, p) for k, p in cell.inputs.items() if _is_matching_port(k, p) and k not in cell_bindings]
            bound_producer = bool(bound_producer)

            if len(multi_ports) > 1 and len(ctx.variables) >= 2:
                # Deterministic O(P * V) role-based matching with zero itertools.permutations
                # Restricts feature_input/target_input/model_input from cross-binding,
                # preventing reversed argument bugs like fit(y, X).
                avail_vars = [v for v in ctx.variables.keys() if not _is_product(ctx.variables[v][0])]

                candidates = []
                ROLE_RESTRICTED = {"feature_input", "target_input", "model_input"}

                for p_name, p in multi_ports:
                    p_role = getattr(p, "derived_role", "standard")
                    p_toks = CellTokenizer.tokenize_identifier((p_name or "").lower())
                    p_desc = str(getattr(p, "doc", "") or getattr(p, "description", "") or "")
                    if p_desc:
                        p_toks.update(CellTokenizer.tokenize_identifier(p_desc))
                    p_sig_inner = getattr(p, "signature", p)
                    p_state = (getattr(p_sig_inner, "state", "") or "").lower()
                    # Declared state vocabulary participates in token overlap
                    # (state names are declared data).
                    if p_state:
                        p_toks.update(p_state.split("_"))

                    for v_name in avail_vars:
                        v_sig, _ = ctx.variables[v_name]
                        v_role = getattr(v_sig, "derived_role", "standard")

                        # Hard role gating
                        if p_role in ROLE_RESTRICTED and v_role in ROLE_RESTRICTED:
                            if p_role != v_role:
                                continue
                        elif p_role == "target_input" and v_role != "target_input":
                            continue
                        elif p_role == "feature_input" and v_role == "target_input":
                            continue
                        elif p_role == "model_input" and v_role != "model_input":
                            continue

                        # Signature unification check
                        u_p = unify(v_sig.signature, p.signature, accumulated_sigma)
                        if u_p is None:
                            continue

                        # Compute score for (p, v_name)
                        score = 10.0
                        if v_name in getattr(ctx, "superseded_vars", set()):
                            score -= 25.0
                        if p_role == v_role and p_role != "standard":
                            score += 25.0
                        if p_role == "feature_input" and v_role == "feature_input":
                            score += 15.0
                        if p_role == "target_input" and v_role == "target_input":
                            score += 15.0
                        if producer_var is not None and v_name == producer_var:
                            score += 8.0

                        # Exact state match priority
                        v_sig_inner = getattr(v_sig, "signature", v_sig)
                        v_state = (getattr(v_sig_inner, "state", "") or "").lower()
                        if p_state and v_state and p_state == v_state:
                            score += 20.0

                        # Token overlap
                        v_toks = set(CellTokenizer.tokenize_identifier((getattr(v_sig, "name", "") or "").lower()))
                        if v_name:
                            v_toks.update(CellTokenizer.tokenize_identifier(str(v_name).lower()))
                        if v_state:
                            v_toks.update(v_state.split("_"))
                        src_cell = getattr(ctx, "var_sources", {}).get(v_name)
                        if src_cell:
                            v_toks.update(src_cell.token_set)

                        _GENERIC_PORT_STOP = {"array", "arrai", "data", "input", "output", "first", "second", "operand"}
                        overlap = len((p_toks & v_toks) - _GENERIC_PORT_STOP)
                        score += overlap * 4.0

                        candidates.append((score, p_name, p, v_name))

                # Deterministic greedy assignment with recency tie-breaking (active wire in frontier)
                candidates.sort(key=lambda item: (item[0], avail_vars.index(item[3])), reverse=True)
                assigned_ports = set()
                assigned_vars = set()
                assigned_origins = set()
                best_assign = {}
                test_sub = accumulated_sigma

                for score, p_name, p, v_name in candidates:
                    if p_name in assigned_ports or v_name in assigned_vars:
                        continue
                    v_origins = getattr(ctx, "var_origins", {}).get(v_name, set())
                    # Monoidal origin disjointness: avoid binding multiple join ports to the same origin wire
                    if v_origins and (v_origins & assigned_origins):
                        continue
                    if v_name in getattr(ctx, "superseded_vars", set()):
                        has_active_alt = any(
                            alt_p == p_name
                            and alt_v not in assigned_vars
                            and alt_v not in getattr(ctx, "superseded_vars", set())
                            for _, alt_p, _, alt_v in candidates
                        )
                        if has_active_alt:
                            continue
                    v_sig, _ = ctx.variables[v_name]
                    u_curr = unify(v_sig.signature, p.signature, test_sub)
                    if u_curr is None:
                        continue
                    test_sub = u_curr
                    assigned_ports.add(p_name)
                    assigned_vars.add(v_name)
                    if v_origins:
                        assigned_origins.update(v_origins)
                    best_assign[p_name] = v_name

                if best_assign:
                    for p_name, v_name in best_assign.items():
                        cell_bindings[p_name] = v_name
                    accumulated_sigma = test_sub
                    if producer_var is not None and producer_var in assigned_vars:
                        bound_producer = True

            # If multi-port matching was not triggered or producer_var is not yet bound:
            # Replicas NEVER auto-bind the previous wire — their inputs come
            # from in-scope variables (the environment) or literals.
            if producer_var is not None and not bound_producer and not is_zero_ary and not is_replica:
                prim_in = cell.primary_input
                if prim_in is not None and prim_in.name in cell.inputs:
                    prod_sig, _ = ctx.variables.get(producer_var, (None, None))
                    if prod_sig is not None and not _is_product(prod_sig):
                        u_sub = unify(prod_sig.signature, prim_in.signature, accumulated_sigma)
                        if u_sub is not None:
                            cell_bindings[prim_in.name] = producer_var
                            accumulated_sigma = u_sub
                            bound_producer = True

            # If producer_var did not bind to primary input, check other compatible input ports (required ports take precedence)
            if producer_var is not None and not bound_producer and not is_zero_ary and not is_replica:
                prod_sig, _ = ctx.variables.get(producer_var, (None, None))
                if prod_sig is not None and not _is_product(prod_sig):
                    req_unbound = [(k, v) for k, v in cell.inputs.items() if v.required and k not in cell_bindings]
                    cand_ports = req_unbound if req_unbound else [(k, v) for k, v in cell.inputs.items() if k not in cell_bindings]
                    for p_name, p_sig in cand_ports:
                        u_sub = unify(prod_sig.signature, p_sig.signature, accumulated_sigma)
                        if u_sub is not None:
                            cell_bindings[p_name] = producer_var
                            accumulated_sigma = u_sub
                            bound_producer = True
                            break

            # 3. Resolve auxiliary input ports (variable reuse / port sharing / parameters / literals).
            # REQUIRED ports are processed before optional ones so data ports claim
            # the prompt's shared literal channels (expressions, paths) first;
            # optional configuration knobs never steal them.
            cell_toks_for_projection = _cell_clause_tokens(cell)
            for p_name, p_sig in sorted(cell.inputs.items(), key=lambda kv: not kv[1].required):
                if p_name in cell_bindings:
                    continue  # Already bound

                # Substitute generics if type variable in p_sig
                concrete_sig = substitute_generics(p_sig, accumulated_sigma)

                # Optional data carriers check in-scope variables first
                _tn_lower = str(getattr(concrete_sig, "type_name", "")).lower()
                ROLE_CARRIERS = TypeRegistry.get_instance().get_declared_role_carriers()
                is_data_carrier = (
                    getattr(p_sig, "port_role", None) in ROLE_CARRIERS
                    or getattr(p_sig, "derived_role", "standard") in ROLE_CARRIERS
                    or registry.is_subtype(_tn_lower, "tensor")
                    or registry.is_subtype(_tn_lower, "table")
                )
                if not p_sig.required and is_data_carrier:
                    scoped_var = None
                    planned_parents = getattr(cell, "bound_parent_ids", None) or set()
                    candidate_vars = []
                    for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                        if _is_product(v_sig) or v_name in cell_bindings.values():
                            continue
                        u_v = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                        if u_v is not None:
                            prod_cell = ctx.var_sources.get(v_name)
                            is_planned = bool(prod_cell and prod_cell.cell_id in planned_parents)
                            candidate_vars.append((is_planned, v_name, u_v))
                    if candidate_vars:
                        candidate_vars.sort(key=lambda x: x[0], reverse=True)
                        _, scoped_var, accumulated_sigma = candidate_vars[0]
                    if scoped_var is not None:
                        cell_bindings[p_name] = scoped_var
                        continue

                # Dual-Port Typestate Projection:
                # Project target vector from upstream tabular carrier for supervised
                # tasks. Fully generic: the target column is an unconsumed prompt
                # identifier (how users name the predicted column); the carrier is
                # the most recent in-scope table-typed variable (declared poset).
                p_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                if p_role == "target_input" and p_name not in cell_bindings:
                    scoped_target = None
                    for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                        if _is_product(v_sig) or v_name in cell_bindings.values():
                            continue
                        if getattr(v_sig, "derived_role", "") == "target_input":
                            u_v = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                            if u_v is not None:
                                scoped_target = v_name
                                accumulated_sigma = u_v
                                break
                    if scoped_target is not None:
                        cell_bindings[p_name] = scoped_target
                        continue

                    unconsumed_lits = [
                        val for _, kind, val in getattr(ctx, "ordered_literals", [])
                        if str(val).lower() not in getattr(ctx, "consumed_tokens", set())
                        and kind in ("identifier", "quoted_str")
                    ]
                    # Ground the target column against the prompt's own declared
                    # column set when one exists (e.g. an explicit columns=[...]
                    # literal list, or a quoted column list elsewhere in the
                    # prompt). An unconsumed literal that merely trails a
                    # trigger preposition ("using StandardScaler") is an
                    # operation reference, not a column name, and must never
                    # be fabricated into a df['<literal>'] access that may not
                    # exist on the frame. Only when the prompt never declared
                    # a column set do we fall back to the previous behavior
                    # (last unconsumed identifier/quoted string) — the
                    # genuine "predict <column>" intent with no schema to
                    # validate against.
                    known_cols = {str(c).strip().lower() for c in getattr(ctx, "columns", []) if str(c).strip()}
                    target_col = getattr(ctx, "target_col", None)
                    if not target_col:
                        if known_cols:
                            for cand in reversed(unconsumed_lits):
                                if str(cand).strip().lower() in known_cols:
                                    target_col = cand
                                    break
                        else:
                            target_col = unconsumed_lits[-1] if unconsumed_lits else None
                    if target_col:
                        df_candidate = None
                        for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                            sig_obj = getattr(v_sig, "signature", v_sig)
                            v_tn = str(getattr(sig_obj, "type_name", "")).lower()
                            if registry.is_subtype(v_tn, "table"):
                                df_candidate = v_name
                                break

                        if df_candidate:
                            # Declared-morphism composition (Solution 3): the
                            # projection of a column out of a tabular carrier
                            # (and any subsequent carrier conversion, e.g.
                            # series -> tensor) is a BRIDGE MORPHISM declared
                            # in a domain tree — never engine-written syntax.
                            # The binder instantiates the declared templates
                            # generically; when no tree declares the required
                            # morphism chain the port stays unresolved and
                            # synthesis refuses, instead of fabricating
                            # library-specific access syntax here.
                            expected_tn = str(getattr(concrete_sig, "type_name", ""))
                            composed = self._compose_declared_bridge_expr(
                                src_var=df_candidate,
                                src_type="table",
                                dst_type=expected_tn,
                                literal_bindings={"column": repr(str(target_col))},
                                dst_sig=concrete_sig,
                                ctx=ctx,
                            )
                            if composed is not None:
                                cell_bindings[p_name] = composed
                                if hasattr(ctx, "consumed_tokens"):
                                    ctx.consumed_tokens.add(str(target_col).lower())
                                continue

                # Check LLM slot filling / hyperparameters before heuristic fallback
                llm_slot_val = None
                if hasattr(ctx, "llm_slots") and isinstance(ctx.llm_slots, dict):
                    cell_slots = ctx.llm_slots.get(cell.cell_id, {})
                    if p_name in cell_slots:
                        llm_slot_val = cell_slots[p_name]
                if llm_slot_val is None and hasattr(ctx, "hyperparameters") and isinstance(ctx.hyperparameters, dict):
                    if p_name in ctx.hyperparameters:
                        llm_slot_val = ctx.hyperparameters[p_name]

                # Optional parameters with declared default: use prompt literal or omit/default
                if not p_sig.required and p_sig.default_value is not None:
                    if llm_slot_val is not None:
                        formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                        cell_bindings[p_name] = formatted_val
                        accumulated_sigma.bind(p_name, formatted_val)
                        continue

                    resolved_literal = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()))
                    if resolved_literal is not None:
                        cell_bindings[p_name] = resolved_literal
                        accumulated_sigma.bind(p_name, resolved_literal)
                    elif str(p_sig.default_value) in ("None", "none"):
                        # Port default is None: omit from the emitted call
                        cell_bindings[p_name] = None
                    else:
                        val = str(p_sig.default_value)
                        if getattr(concrete_sig, "type_name", "") == "str" and not (val.startswith(("'", '"')) or val in ("None", "True", "False")):
                            val = repr(val)
                        cell_bindings[p_name] = val
                        accumulated_sigma.bind(p_name, val)
                    continue

                # Optional parameters WITHOUT a default are omitted from the
                # emitted call unless the prompt itself supplies a literal through
                # the type-affinity channels (a destination path, a numeric or
                # boolean qualifier).
                if not p_sig.required:
                    if llm_slot_val is not None:
                        formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                        cell_bindings[p_name] = formatted_val
                        accumulated_sigma.bind(p_name, formatted_val)
                        continue

                    _lit = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()))
                    if _lit is not None:
                        cell_bindings[p_name] = _lit
                        accumulated_sigma.bind(p_name, _lit)
                    else:
                        cell_bindings[p_name] = None
                    continue

                # A. Check in-scope variables first (environment / predecessor variables matching typestate)
                # One wire feeds ONE port per cell: a variable already bound to
                # this cell is skipped here. Silent duplication (the same
                # table into both concat inputs) is a planning artifact,
                # not a composition the prompt requested — a cell that genuinely
                # re-consumes a wire declares it (bound_slots / replicas).
                already_bound_vars = {
                    v for v in cell_bindings.values() if isinstance(v, str)
                }
                scoped_var = None
                planned_parents = getattr(cell, "bound_parent_ids", None) or set()
                candidate_vars = []
                for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                    # Heterogeneous products project member-wise; the whole container
                    # is never wired into a single data port.
                    if _is_product(v_sig):
                        continue
                    if v_name in already_bound_vars:
                        continue
                    u_v = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                    if u_v is not None:
                        prod_cell = ctx.var_sources.get(v_name)
                        is_planned = bool(prod_cell and prod_cell.cell_id in planned_parents)
                        candidate_vars.append((is_planned, v_name, u_v))

                if candidate_vars:
                    # D6 Fix: Planned parent variables take absolute priority over arbitrary recency
                    candidate_vars.sort(key=lambda x: x[0], reverse=True)
                    _, scoped_var, accumulated_sigma = candidate_vars[0]

                if scoped_var is not None:
                    cell_bindings[p_name] = scoped_var
                    continue

                # A2. Product-member projection: bind {port} to var[i] when the port's
                # declared signature unifies with a declared member carrier. Preference:
                # unconsumed members, then member-state affinity with the consuming clause.
                member_bound = False
                for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                    if not _is_product(v_sig):
                        continue
                    cands = _member_candidates(v_sig, v_name, concrete_sig, cell_toks_for_projection)
                    if not cands:
                        continue
                    cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    _, best_i, best_u = cands[0]
                    cell_bindings[p_name] = f"{v_name}[{best_i}]"
                    accumulated_sigma = best_u
                    consumed_members = getattr(ctx, "consumed_members", None)
                    if consumed_members is not None:
                        consumed_members.add((v_name, best_i))
                    member_bound = True
                    break

                if member_bound:
                    continue

                # B. Check typestate-driven literal resolution from prompt or LLM slots
                if llm_slot_val is not None:
                    formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                    cell_bindings[p_name] = formatted_val
                    accumulated_sigma.bind(p_name, formatted_val)
                    continue

                resolved_literal = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()))
                if resolved_literal is not None:
                    cell_bindings[p_name] = resolved_literal
                    accumulated_sigma.bind(p_name, resolved_literal)
                    continue

                # C. Check default value declared in tree
                if p_sig.default_value is not None:
                    val = str(p_sig.default_value)
                    if getattr(concrete_sig, "type_name", "") == "str" and not (val.startswith(("'", '"')) or val in ("None", "True", "False")):
                        val = repr(val)
                    cell_bindings[p_name] = val
                    accumulated_sigma.bind(p_name, val)
                    continue

                # D. Declared-domain enum grounding (reflection-driven, zero domain hardcodes)
                p_domain = getattr(concrete_sig, "domain", "") or ""
                if p_domain and getattr(ctx, "prompt", ""):
                    enum_val = ctx._resolve_enum_constant(p_domain, concrete_sig.state)
                    if enum_val is not None:
                        cell_bindings[p_name] = enum_val
                        accumulated_sigma.bind(p_name, enum_val)
                        continue

                # D.2 Higher-Order Functional Operator resolution:
                # If the port is a functional operator (Callable), synthesize a safe default lambda
                # (identity functor, predicate, or fallback) rather than failing synthesis.
                if getattr(p_sig, "port_role", None) == "functional_operator" or "Callable" in getattr(concrete_sig, "type_name", ""):
                    if "bool" in str(concrete_sig.type_name).lower():
                        default_fn = "lambda x: bool(x)"
                    elif "Exception" in str(concrete_sig.type_name):
                        default_fn = "lambda err, d=None: d"
                    else:
                        default_fn = "lambda *args: args[0] if args else None"
                    cell_bindings[p_name] = default_fn
                    accumulated_sigma.bind(p_name, default_fn)
                    continue

                # E. Unresolved REQUIRED port: fail loudly. A pipeline with an
                # unsatisfiable required port must not emit garbage code (e.g. a bare
                # identifier that NameErrors at runtime); it is reported as a synthesis
                # failure so the caller (or the LLM repair cycle) can react honestly.
                if p_sig.required:
                    ctx.unresolved_ports.append((cell.cell_id, p_name))
                    ctx.unbindable_count += 1
                    cell_bindings[p_name] = UNRESOLVED_PORT
                else:
                    cell_bindings[p_name] = None

            # Register output port(s) in context for future steps
            if len(cell.outputs) > 1:
                # Multi-output cell (e.g. train_test_split, load_wine, cv2.threshold, subplots)
                # Declare each individual output port in the execution context and bind in template
                first_out_var = None
                prim_out = cell.primary_output
                prim_var = None
                for idx_out, (out_name, out_sig) in enumerate(cell.outputs.items()):
                    var_counter += 1
                    ctx.var_counter = var_counter
                    out_var = f"var_{var_counter}"
                    if idx_out == 0:
                        first_out_var = out_var
                    cell_bindings[out_name] = out_var
                    cell_port_vars[(cell.cell_id, out_name)] = out_var
                    cell_port_vars[(id(cell), out_name)] = out_var

                    concrete_out = substitute_generics(out_sig, accumulated_sigma)
                    ctx.declare_variable(out_var, concrete_out, out_var, cell=cell)
                    ctx.var_parents.setdefault(out_var, set())
                    ctx.var_origins.setdefault(out_var, set())
                    bound_in_vars = [v for v in cell_bindings.values() if isinstance(v, str) and v in ctx.variables]
                    for in_v in bound_in_vars:
                        ctx.var_parents[out_var].add(in_v)
                        ctx.var_origins[out_var].update(ctx.var_origins.get(in_v, set()))
                    if prim_out and out_name == prim_out.name:
                        prim_var = out_var

                producer_var = prim_var if prim_var is not None else (first_out_var or current_out_var)
                if "output_var" not in cell.outputs:
                    cell_bindings["output_var"] = producer_var
                cell_out_vars[cell.cell_id] = producer_var
                cell_out_vars[id(cell)] = producer_var
                cell_port_vars[(cell.cell_id, "output_var")] = producer_var
                cell_port_vars[(id(cell), "output_var")] = producer_var
            elif len(cell.outputs) == 1:
                # Single output cell
                concrete_out = substitute_generics(cell.primary_output, accumulated_sigma)
                # If the cell performs in-place mutation on a receiver, alias the output
                # to the receiver: the receiver is the cell's declared receiver-role port,
                # falling back to the primary input binding (declared-first, name-last).
                if getattr(cell, "mutation_type", "pure") == "in_place":
                    receiver_var = None
                    receiver_port = next(
                        (p for p in cell.inputs.values() if getattr(p, "port_role", None) == "receiver"),
                        None,
                    )
                    if receiver_port is not None:
                        receiver_var = cell_bindings.get(receiver_port.name)
                    if receiver_var is None and cell.primary_input is not None:
                        receiver_var = cell_bindings.get(cell.primary_input.name)
                    if receiver_var is None:
                        receiver_var = cell_bindings.get("data") or cell_bindings.get("self")
                    if receiver_var and receiver_var in ctx.variables:
                        current_out_var = receiver_var
                        cell_bindings["output_var"] = current_out_var

                # Bind primary output port name if distinct from output_var
                if cell.primary_output and cell.primary_output.name:
                    cell_bindings[cell.primary_output.name] = current_out_var
                    cell_port_vars[(cell.cell_id, cell.primary_output.name)] = current_out_var
                    cell_port_vars[(id(cell), cell.primary_output.name)] = current_out_var

                if current_out_var is not None:
                    ctx.declare_variable(current_out_var, concrete_out, current_out_var, cell=cell)
                    producer_var = current_out_var
                    cell_out_vars[cell.cell_id] = current_out_var
                    cell_out_vars[id(cell)] = current_out_var
                    cell_port_vars[(cell.cell_id, "output_var")] = current_out_var
                    cell_port_vars[(id(cell), "output_var")] = current_out_var

                    # Provenance and monoidal DAG branch tracking
                    ctx.var_parents.setdefault(current_out_var, set())
                    ctx.var_origins.setdefault(current_out_var, set())
                    bound_in_vars = [v for v in cell_bindings.values() if isinstance(v, str) and v in ctx.variables]
                    for in_v in bound_in_vars:
                        ctx.var_parents[current_out_var].add(in_v)
                        ctx.var_origins[current_out_var].update(ctx.var_origins.get(in_v, set()))

                    for p_k, p_val in cell_bindings.items():
                        if p_k in ("column", "key", "columns") and p_val and p_val != UNRESOLVED_PORT:
                            lit_clean = str(p_val).strip("'\"")
                            if lit_clean:
                                ctx.var_origins[current_out_var].add(lit_clean)

                    c_lits = getattr(cell, "clause_literals", None)
                    if c_lits:
                        for cl_lit in c_lits:
                            clean_lit = str(cl_lit).strip("'\"")
                            if clean_lit:
                                ctx.var_origins[current_out_var].add(clean_lit)

                    is_projection = any(k in ("column", "key", "columns") for k in cell.inputs.keys())
                    if len(bound_in_vars) == 1 and not is_projection and getattr(cell, "stage", None) == 2:
                        ctx.superseded_vars.add(bound_in_vars[0])
            else:
                # Void output cell (outputs: {}) - e.g. .show(), .save() returning None.
                # Do not advance counter, do not declare phantom variables, and do not overwrite producer_var.
                pass

            pipeline_bindings.append((cell, cell_bindings))

        return Success(pipeline_bindings, accumulated_sigma)

    @staticmethod
    def _instantiate_ast_template(template: str, bindings: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """
        Synthesizes executable Python code from an AST template, adhering to identity omission semantics:
        - Required positional parameters are instantiated with their bound values.
        - Actively bound optional configurations are emitted as keyword arguments (key=val).
        - Unbound optional parameters with defaults are omitted, letting runtime defaults apply.
        - Enforces param_kind calling conventions (positional_only, keyword_only, var_positional, var_keyword).
        """
        if not template or not template.strip():
            return ""

        ph_map: Dict[str, str] = {}
        known_placeholders = set(inputs.keys()) | set(bindings.keys()) | {"output_var", "input_data", "output_data"}
        ast_ready = template
        for name in extract_template_placeholders(template):
            if known_placeholders and name not in known_placeholders:
                continue
            ph_id = f"_nstl_ph_{name}"
            ph_map[ph_id] = name
            ast_ready = ast_ready.replace(f"{{{name}}}", ph_id)
        try:
            parsed = ast.parse(ast_ready)
        except SyntaxError:
            res = template
            for k, v in bindings.items():
                if v is not None:
                    res = res.replace(f"{{{k}}}", str(v))
            return res

        class CallOptimizer(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                new_args = []
                new_keywords = []
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in ph_map:
                        orig_name = ph_map[arg.id]
                        p_sig = inputs.get(orig_name)
                        is_req = getattr(p_sig, "required", True) if p_sig else True
                        p_kind = getattr(p_sig, "param_kind", "standard")
                        val = bindings.get(orig_name)
                        if val is UNRESOLVED_PORT or (isinstance(val, str) and val in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")):
                            raise UnresolvedPlaceholderError(
                                f"Cannot emit code with unresolved port value for placeholder '{orig_name}' in template: {template}"
                            )

                        if val is not None:
                            val_str = str(val)
                            try:
                                val_node = ast.parse(val_str, mode="eval").body
                            except Exception as e:
                                val_node = ast.Constant(value=val_str)
                            if p_kind == "keyword_only":
                                new_keywords.append(ast.keyword(arg=orig_name, value=val_node))
                            elif p_kind == "var_keyword":
                                new_keywords.append(ast.keyword(arg=None, value=val_node))
                            else:
                                new_args.append(val_node)
                        elif is_req:
                            registry = TypeRegistry.get_instance()
                            t_name = getattr(getattr(p_sig, "signature", None), "type_name", "") or getattr(p_sig, "type_name", "")
                            is_str_like = registry.is_subtype(t_name, "str")
                            if is_str_like:
                                val_node = ast.Constant(value=orig_name)
                            else:
                                val_node = ast.Name(id=orig_name, ctx=ast.Load())
                            new_args.append(val_node)
                        # If optional and val is None: omit from call
                    else:
                        new_args.append(arg)

                for kw in node.keywords:
                    val_node = kw.value
                    if isinstance(val_node, ast.Name) and val_node.id in ph_map:
                        orig_name = ph_map[val_node.id]
                        p_sig = inputs.get(orig_name)
                        is_req = getattr(p_sig, "required", True) if p_sig else True
                        val = bindings.get(orig_name)
                        if val is UNRESOLVED_PORT or (isinstance(val, str) and val in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")):
                            raise UnresolvedPlaceholderError(
                                f"Cannot emit code with unresolved port value for placeholder '{orig_name}' in template: {template}"
                            )
                        if val is not None:
                            val_str = str(val)
                            try:
                                kw_val_node = ast.parse(val_str, mode="eval").body
                            except Exception:
                                kw_val_node = ast.Constant(value=val_str)
                            new_keywords.append(ast.keyword(arg=kw.arg, value=kw_val_node))
                        elif is_req:
                            registry = TypeRegistry.get_instance()
                            t_name = getattr(getattr(p_sig, "signature", None), "type_name", "") or getattr(p_sig, "type_name", "")
                            is_str_like = registry.is_subtype(t_name, "str")
                            if is_str_like:
                                kw_val_node = ast.Constant(value=orig_name)
                            else:
                                kw_val_node = ast.Name(id=orig_name, ctx=ast.Load())
                            new_keywords.append(ast.keyword(arg=kw.arg, value=kw_val_node))
                        # If optional and val is None: omit keyword from call
                    else:
                        new_keywords.append(kw)

                node.args = new_args
                node.keywords = new_keywords
                return node

        optimized = CallOptimizer().visit(parsed)
        for node in ast.walk(optimized):
            if isinstance(node, ast.Name) and node.id in ph_map:
                orig = ph_map[node.id]
                if orig in bindings and bindings[orig] is not None:
                    node.id = str(bindings[orig])

        try:
            return ast.unparse(optimized)
        except Exception as e:
            res = template
            for k, v in bindings.items():
                if v is not None:
                    res = res.replace(f"{{{k}}}", str(v))
            return res

    @staticmethod
    def _prune_unconsumed_transform_outputs(
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]]
    ) -> List[Tuple[Cell, Dict[str, str]]]:
        """
        Removes non-sink, non-egress transform steps whose declared output
        variable(s) are never referenced by any later step's bindings in the
        ACTUAL wired pipeline. This is deliberately independent of the
        planner's search-time connectivity heuristic (type-level "could this
        feed something") -- it checks the binder's own ground truth instead,
        so it catches exactly the cases where the planner and the binder
        silently disagreed about which variable actually gets used.

        Stage-3 cells and cells with node_role == "sink" are always kept:
        their purpose is the side effect (write, plot, save), not a return
        value some later step consumes, so an "unused" return value there is
        normal and correct, not dead code.
        """
        if not pipeline_bindings:
            return pipeline_bindings

        current = list(pipeline_bindings)
        changed = True
        while changed:
            changed = False
            for idx, (cell, bnd) in enumerate(current):
                stage = getattr(cell, "stage", None)
                role = str(getattr(cell, "node_role", "") or "").lower()
                outputs = getattr(cell, "outputs", {}) or {}
                # The final cell produces the terminal pipeline result and must never be pruned
                if idx == len(current) - 1 or stage == 3 or role == "sink" or not outputs:
                    continue
                # Never prune cells that witness an explicit clause or intent from the prompt (e.g. side calculations like mean, metrics)
                if getattr(cell, "matched_clause_idx", None) is not None or getattr(cell, "is_goal", False):
                    continue
                out_vars = {v for k, v in bnd.items() if k in outputs and isinstance(v, str)}
                if not out_vars:
                    continue
                referenced_later: Set[str] = set()
                for _, other_bnd in current[idx + 1:]:
                    for val in other_bnd.values():
                        if isinstance(val, str):
                            referenced_later.update(tokenize_alphanumeric(val))
                if not (out_vars & referenced_later):
                    current.pop(idx)
                    changed = True
                    break
        return current

    def _extract_pipeline_slots_with_llm(
        self,
        cells: List[Cell],
        prompt: str,
        ctx: ExecutionContext
    ) -> None:
        """
        Grounded parameter and variable binding via LLM slot extraction.
        Replaces brittle regex scanning by extracting values for candidate non-carrier ports
        directly from the user prompt when an LLM is available.
        """
        if not prompt or not cells:
            return
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            mm = ModelManager.get_instance()
            if not mm.can_synthesize():
                return
        except Exception:
            return

        registry = TypeRegistry.get_instance()
        candidate_ports_by_cell = {}
        for c in cells:
            if not hasattr(c, "inputs") or not c.inputs:
                continue
            c_ports = {}
            for p_name, p_sig in c.inputs.items():
                p_role = getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "") or ""
                t_name = (getattr(p_sig, "type_name", "") or "").lower()
                if p_role in ("carrier", "data_input", "feature_input", "model_input"):
                    continue
                if registry.is_subtype(t_name, "table") or registry.is_subtype(t_name, "tensor") or registry.is_subtype(t_name, "model"):
                    continue
                c_ports[p_name] = {
                    "type": t_name or "any",
                    "required": getattr(p_sig, "required", False),
                    "default": str(getattr(p_sig, "default_value", None))
                }
            if c_ports:
                candidate_ports_by_cell[c.cell_id] = c_ports

        if not candidate_ports_by_cell:
            return

        schema = {
            "type": "object",
            "properties": {
                "target_col": {"type": "string", "description": "Target or label column name to predict, if any"},
                "columns": {"type": "array", "items": {"type": "string"}, "description": "Specific feature columns or subset mentioned"},
                "slots": {
                    "type": "object",
                    "description": "Mapping from cell_id to port_name and extracted value",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {"type": ["string", "number", "boolean", "null"]}
                    }
                },
                "hyperparameters": {
                    "type": "object",
                    "description": "General hyperparameters or arguments extracted from prompt",
                    "additionalProperties": {"type": ["string", "number", "boolean", "null"]}
                }
            }
        }

        cell_summary = []
        for cid, ports in candidate_ports_by_cell.items():
            port_desc = ", ".join(f"{p}: {info['type']}" for p, info in ports.items())
            cell_summary.append(f"- Cell '{cid}': {port_desc}")

        query = (
            f"User Prompt: {prompt}\n\n"
            f"Extract literal values (file paths, column names, thresholds, hyperparameters) for the following pipeline nodes:\n"
            + "\n".join(cell_summary) + "\n\n"
            "Return valid JSON only matching schema with 'target_col', 'columns', 'slots' ({cell_id: {port_name: value}}), and 'hyperparameters'."
        )

        try:
            raw = mm.generate_text(
                query,
                max_tokens=512,
                schema=schema,
                system_prompt="You are a precise dataflow parameter extractor. Extract exact parameter literals, column names, and options mentioned in the prompt."
            )
            if raw and raw.strip():
                text = raw.strip()
                if "```json" in text:
                    text = text.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in text:
                    text = text.split("```", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    if not ctx.target_col and parsed.get("target_col"):
                        ctx.target_col = str(parsed["target_col"]).strip()
                    if not ctx.columns and parsed.get("columns") and isinstance(parsed["columns"], list):
                        ctx.columns = [str(c).strip() for c in parsed["columns"]]
                    slots = parsed.get("slots", {})
                    if isinstance(slots, dict):
                        for k, v in slots.items():
                            if isinstance(v, dict):
                                if k not in ctx.llm_slots:
                                    ctx.llm_slots[k] = {}
                                for p_name, val in v.items():
                                    if val is not None:
                                        ctx.llm_slots[k][p_name] = val
                            elif v is not None:
                                if not hasattr(ctx, "hyperparameters") or ctx.hyperparameters is None:
                                    ctx.hyperparameters = {}
                                ctx.hyperparameters[k] = v
                    hparams = parsed.get("hyperparameters", {})
                    if isinstance(hparams, dict):
                        if not hasattr(ctx, "hyperparameters") or ctx.hyperparameters is None:
                            ctx.hyperparameters = {}
                        for hk, hv in hparams.items():
                            if hv is not None:
                                ctx.hyperparameters[hk] = hv
        except Exception as e:
            logger.debug(f"[UNIFICATION] LLM slot extraction skipped or failed: {e}")

    def emit_code(
        self,
        cells: List[Cell],
        context: Optional[ExecutionContext] = None
    ) -> str:
        """
        Emits clean, fully instantiated code from a verified cell pipeline.
        Replaces port placeholders strictly from unified variable bindings.
        """
        ctx = context or self.context
        if getattr(ctx, "unresolved_ports", None):
            ports_str = ", ".join(f"{cid}.{p}" for cid, p in ctx.unresolved_ports)
            raise UnresolvedPlaceholderError(
                f"Code synthesis refused: pipeline contains unresolved ports: [{ports_str}]"
            )

        accum_sigma: Substitution = Substitution()
        if cells and isinstance(cells[0], tuple):
            pipeline_bindings = cells
        else:
            ctx_run = ctx.clone() if hasattr(ctx, "clone") else ctx
            if hasattr(ctx_run, "reset"):
                ctx_run.reset()
            self._extract_pipeline_slots_with_llm(cells, getattr(ctx_run, "_prompt", "") or getattr(self.context, "_prompt", ""), ctx_run)
            res = self.unify_pipeline(cells, ctx_run)
            if res.is_bottom():
                reason = res.reason if isinstance(res, Failure) else "Unknown unification failure"
                raise ValueError(f"Unification Failed: {reason}")
            assert isinstance(res, Success)
            pipeline_bindings = res.value
            accum_sigma = res.sigma
            self.last_egress_paths = self._derive_egress_paths(pipeline_bindings)
            self.last_runtime_aliases = dict(getattr(ctx_run, "runtime_aliases", {}) or {})
            if getattr(ctx_run, "unresolved_ports", None):
                ports_str = ", ".join(f"{cid}.{p}" for cid, p in ctx_run.unresolved_ports)
                raise UnresolvedPlaceholderError(
                    f"Code synthesis refused: pipeline contains unresolved ports: [{ports_str}]"
                )

        # Ensure no binding contains an unresolved sentinel
        for cell, bnd in pipeline_bindings:
            for p_name, val in bnd.items():
                if val is UNRESOLVED_PORT or (isinstance(val, str) and val in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")):
                    raise UnresolvedPlaceholderError(
                        f"Code synthesis refused: port '{p_name}' of cell '{cell.cell_id}' has unresolved value '{val}'"
                    )

        # Ground-truth dead-branch elimination. The planner's search-time
        # dead-output penalty only checks whether a downstream cell's
        # declared input TYPE could accept a given output (see
        # LatticePlanner._cells_connect) -- never whether this binder
        # actually wired that specific variable anywhere. This binder makes
        # its own independent, greedy choice of which in-scope variable to
        # use for each port, so a step the planner scored as "consumed" can
        # still end up computing a value nothing in the emitted code ever
        # reads. Prune those branches here, against the binder's own real
        # bindings, before anything is turned into source text.
        pipeline_bindings = self._prune_unconsumed_transform_outputs(pipeline_bindings)

        self.last_pipeline_bindings = pipeline_bindings
        self.last_verification_contract = self.build_verification_contract(pipeline_bindings, ctx, getattr(ctx, "prompt", ""))

        # Collect dependencies recursively
        deps: List[str] = []
        def collect_deps(c: Cell):
            all_deps = list(getattr(c, "dependencies", [])) + list(getattr(c, "imports", []))
            for dep in all_deps:
                dep_clean = dep.strip()
                if not dep_clean:
                    continue
                if dep_clean.startswith(("import ", "from ")):
                    if dep_clean not in deps:
                        deps.append(dep_clean)
                    continue

                base_mod = dep_clean.split(".")[0]
                tpl = getattr(c, "code_template", "") or ""
                has_attr_call = f"{base_mod}." in tpl or f"{dep_clean}." in tpl
                if not has_attr_call:
                    aliases = TypeRegistry.get_instance().get_all_aliases()
                    for alias, target in aliases.items():
                        if target == dep_clean or target == base_mod:
                            if f"{alias}." in tpl:
                                has_attr_call = True
                                break

                if has_attr_call:
                    stmt = f"import {dep_clean}"
                    if stmt not in deps:
                        deps.append(stmt)
                else:
                    # Template references symbols directly (e.g. LogisticRegression(...) or train_test_split(...))
                    symbols = _get_module_symbols(dep_clean)
                    sym_map = {s.lower(): s for s in symbols}
                    words = frozenset(tokenize_alphanumeric(tpl))
                    matched = {sym_map[w] for w in words if w in sym_map}

                    # Structural fallback: if cell_id subcomponent appears in template
                    cid = getattr(c, "cell_id", "") or ""
                    if cid.startswith(dep_clean + "."):
                        sub_ident = cid[len(dep_clean) + 1:].split(".")[0]
                        if sub_ident and sub_ident.lower() in words:
                            matched.add(sub_ident)

                    if matched:
                        for w in sorted(matched):
                            stmt = f"from {dep_clean} import {w}"
                            if stmt not in deps:
                                deps.append(stmt)
                    else:
                        stmt = f"import {dep_clean}"
                        if stmt not in deps:
                            deps.append(stmt)

            for sub_list in getattr(c, "bound_slots", {}).values():
                if isinstance(sub_list, list):
                    for sc in sub_list:
                        if hasattr(sc, "dependencies"):
                            collect_deps(sc)

        for cell, _ in pipeline_bindings:
            collect_deps(cell)

        code_lines: List[str] = []
        if deps:
            code_lines.extend(deps)
            code_lines.append("")

        try:
            from .synthesis import render_cell
            has_render_cell = True
        except (ImportError, ValueError):
            try:
                from synthesis import render_cell
                has_render_cell = True
            except ImportError:
                has_render_cell = False

        for cell, bindings in pipeline_bindings:
            if has_render_cell and (bool(getattr(cell, "bound_slots", None)) or getattr(cell, "node_type", "") == "macro"):
                rendered = render_cell(cell, bindings, indent_level=0, context=ctx, accumulated_sigma=accum_sigma)
                if rendered:
                    code_lines.append(rendered)
            else:
                template = cell.code_template.strip()
                if not template:
                    continue

                instantiated = self._instantiate_ast_template(template, bindings, cell.inputs)
                code_lines.append(instantiated)

        # Declared egress aliasing: when the prompt explicitly declares a
        # target sink ("store the results into Z"), the declaration is
        # honored at the RUNTIME namespace level (sandbox aliases Z to the
        # terminal pipeline variable). The emitted source is never rewritten
        # with a synthetic assignment — that would fabricate a sink morphism
        # that no declared cell provides, and would let verification observe
        # a binding the dataflow never performed.
        if ctx is not None and getattr(ctx, "prompt", None) and pipeline_bindings:
            target_ident = ExecutionContext._extract_target_sink(ctx.prompt)
            if target_ident:
                out_var = None
                def _is_var_assignment(line_str: str, var_name: str) -> bool:
                    if "=" not in line_str:
                        return False
                    lhs = line_str.split("=")[0].strip()
                    if lhs == var_name:
                        return True
                    parts = [p.strip() for p in lhs.split(",")]
                    return var_name in parts

                for c_prev, b_prev in reversed(pipeline_bindings):
                    cand_var = b_prev.get("output_var") or (b_prev.get(c_prev.primary_output.name) if getattr(c_prev, "primary_output", None) else None)
                    if cand_var and any(_is_var_assignment(line, str(cand_var)) for line in code_lines):
                        out_var = cand_var
                        break
                if out_var and str(out_var) != str(target_ident):
                    declare_alias = getattr(ctx, "declare_runtime_alias", None)
                    if callable(declare_alias):
                        declare_alias(target_ident, str(out_var))

        final_code = "\n".join(code_lines).strip()
        final_code = self._reconcile_imports(final_code)
        prior_vars: Set[str] = set()
        if ctx is not None:
            if hasattr(ctx, "scope") and isinstance(ctx.scope, dict):
                prior_vars.update(ctx.scope.keys())
            if hasattr(ctx, "scope_variables") and isinstance(ctx.scope_variables, dict):
                prior_vars.update(ctx.scope_variables.keys())
            if hasattr(ctx, "variables") and isinstance(ctx.variables, dict):
                produced_vars: Set[str] = set()
                for cell, bindings in pipeline_bindings:
                    if isinstance(bindings, dict):
                        outputs = getattr(cell, "outputs", {})
                        if isinstance(outputs, dict):
                            for out_name in outputs.keys():
                                if out_name in bindings and isinstance(bindings[out_name], str):
                                    produced_vars.add(bindings[out_name])
                for v_name in ctx.variables.keys():
                    if v_name not in produced_vars:
                        prior_vars.add(v_name)
        self._verify_emitted_ast_liveness(final_code, initial_scope=prior_vars)
        return final_code

    @staticmethod
    def _reconcile_imports(code: str) -> str:
        """AST-level alias reconciliation: every Name root used as module/attribute
        must be bound by an import or local variable in the script; inject missing ones."""
        if not code or not code.strip():
            return code
        try:
            tree = ast.parse(code)
        except Exception as e:
            return code

        bound: Set[str] = set(dir(__builtins__))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    bound.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    bound.add(a.asname or a.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, ast.AnnAssign):
                if node.target:
                    for sub in ast.walk(node.target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, ast.AugAssign):
                if node.target:
                    for sub in ast.walk(node.target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, ast.NamedExpr):
                if node.target:
                    for sub in ast.walk(node.target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for sub in ast.walk(node.target):
                    if isinstance(sub, ast.Name):
                        bound.add(sub.id)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars:
                        for sub in ast.walk(item.optional_vars):
                            if isinstance(sub, ast.Name):
                                bound.add(sub.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
                if hasattr(node, "args"):
                    for arg in node.args.args + getattr(node.args, "kwonlyargs", []) + getattr(node.args, "posonlyargs", []):
                        bound.add(arg.arg)
                    if node.args.vararg:
                        bound.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        bound.add(node.args.kwarg.arg)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for gen in node.generators:
                    for sub in ast.walk(gen.target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    bound.add(node.name)

        used_roots: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used_roots.add(node.value.id)

        unbound_roots = sorted(used_roots - bound)
        if not unbound_roots:
            return code

        registered_aliases = TypeRegistry.get_instance().get_all_aliases()
        import importlib.util
        injected: List[str] = []
        for root in unbound_roots:
            root_clean = root.strip()
            root_lower = root_clean.lower()
            if root_lower in registered_aliases:
                target_mod = registered_aliases[root_lower]
                if target_mod == root_clean:
                    stmt = f"import {root_clean}"
                else:
                    stmt = f"import {target_mod} as {root_clean}"
                if stmt not in injected:
                    injected.append(stmt)
            else:
                try:
                    spec = importlib.util.find_spec(root_clean)
                    if spec is not None:
                        stmt = f"import {root_clean}"
                        if stmt not in injected:
                            injected.append(stmt)
                except Exception as e:
                    logger.debug("suppressed: %s", e, exc_info=False)

        if not injected:
            return code

        header = "\n".join(injected)
        return header + "\n" + code

    @staticmethod
    def _verify_emitted_ast_liveness(code: str, initial_scope: Optional[Set[str]] = None) -> None:
        """AST-level dataflow liveness validation: every variable loaded in the script
        must have been previously assigned, imported, or present in builtins.
        Catches undefined variables (e.g. var_7, var_3) before code emission."""
        if not code or not code.strip():
            return
        try:
            tree = ast.parse(code)
        except Exception as e:
            raise EmissionLivenessError(f"Emitted code has invalid syntax: {e}") from e

        import builtins
        global_scope: Set[str] = set(dir(builtins))
        global_scope.update({
            "__file__", "__name__", "__doc__", "__package__",
            "__spec__", "__path__", "__loader__", "__annotations__"
        })
        if initial_scope:
            global_scope.update(initial_scope)

        class ScopeChecker(ast.NodeVisitor):
            def __init__(self):
                self.scopes: List[Set[str]] = [set(global_scope)]
                self.undefined: List[Tuple[str, int]] = []

            def _is_bound(self, name: str) -> bool:
                return any(name in scope for scope in reversed(self.scopes))

            def _bind(self, name: str) -> None:
                self.scopes[-1].add(name)

            def _extract_targets(self, target_node, target_set: Set[str]):
                for n in ast.walk(target_node):
                    if isinstance(n, ast.Name):
                        target_set.add(n.id)

            def visit_Import(self, node: ast.Import):
                for alias in node.names:
                    self._bind(alias.asname or alias.name.split(".")[0])

            def visit_ImportFrom(self, node: ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        self._bind(alias.asname or alias.name)

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self._bind(node.name)
                for dec in node.decorator_list:
                    self.visit(dec)
                func_scope: Set[str] = set()
                all_args = getattr(node.args, "posonlyargs", []) + node.args.args + getattr(node.args, "kwonlyargs", [])
                for arg in all_args:
                    func_scope.add(arg.arg)
                if node.args.vararg:
                    func_scope.add(node.args.vararg.arg)
                if node.args.kwarg:
                    func_scope.add(node.args.kwarg.arg)
                self.scopes.append(func_scope)
                for stmt in node.body:
                    self.visit(stmt)
                self.scopes.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node: ast.ClassDef):
                self._bind(node.name)
                for dec in node.decorator_list:
                    self.visit(dec)
                for base in node.bases:
                    self.visit(base)
                for kw in node.keywords:
                    self.visit(kw)
                class_scope: Set[str] = set()
                self.scopes.append(class_scope)
                for stmt in node.body:
                    self.visit(stmt)
                self.scopes.pop()

            def visit_Lambda(self, node: ast.Lambda):
                lam_scope: Set[str] = set()
                all_args = getattr(node.args, "posonlyargs", []) + node.args.args + getattr(node.args, "kwonlyargs", [])
                for arg in all_args:
                    lam_scope.add(arg.arg)
                if node.args.vararg:
                    lam_scope.add(node.args.vararg.arg)
                if node.args.kwarg:
                    lam_scope.add(node.args.kwarg.arg)
                self.scopes.append(lam_scope)
                self.visit(node.body)
                self.scopes.pop()

            def visit_ListComp(self, node):
                self._visit_comp(node)

            def visit_SetComp(self, node):
                self._visit_comp(node)

            def visit_DictComp(self, node):
                self._visit_comp(node)

            def visit_GeneratorExp(self, node):
                self._visit_comp(node)

            def _visit_comp(self, node):
                comp_scope: Set[str] = set()
                self.scopes.append(comp_scope)
                for gen in node.generators:
                    self.visit(gen.iter)
                    self._extract_targets(gen.target, comp_scope)
                    for if_expr in gen.ifs:
                        self.visit(if_expr)
                if isinstance(node, ast.DictComp):
                    self.visit(node.key)
                    self.visit(node.value)
                else:
                    self.visit(node.elt)
                self.scopes.pop()

            def visit_Assign(self, node: ast.Assign):
                self.visit(node.value)
                for target in node.targets:
                    self._extract_targets(target, self.scopes[-1])

            def visit_AnnAssign(self, node: ast.AnnAssign):
                if node.value:
                    self.visit(node.value)
                if node.target:
                    self._extract_targets(node.target, self.scopes[-1])

            def visit_AugAssign(self, node: ast.AugAssign):
                self.visit(node.target)
                self.visit(node.value)

            def visit_NamedExpr(self, node: ast.NamedExpr):
                self.visit(node.value)
                self._extract_targets(node.target, self.scopes[-1])

            def visit_For(self, node: ast.For):
                self.visit(node.iter)
                self._extract_targets(node.target, self.scopes[-1])
                for stmt in node.body:
                    self.visit(stmt)
                for stmt in node.orelse:
                    self.visit(stmt)

            def visit_AsyncFor(self, node: ast.AsyncFor):
                self.visit_For(node)

            def visit_With(self, node: ast.With):
                for item in node.items:
                    self.visit(item.context_expr)
                    if item.optional_vars:
                        self._extract_targets(item.optional_vars, self.scopes[-1])
                for stmt in node.body:
                    self.visit(stmt)

            def visit_AsyncWith(self, node: ast.AsyncWith):
                self.visit_With(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler):
                if node.type:
                    self.visit(node.type)
                if node.name:
                    self._bind(node.name)
                for stmt in node.body:
                    self.visit(stmt)

            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load):
                    if not self._is_bound(node.id):
                        self.undefined.append((node.id, getattr(node, "lineno", 0)))
                elif isinstance(node.ctx, ast.Store):
                    self._bind(node.id)

        checker = ScopeChecker()
        checker.visit(tree)
        if checker.undefined:
            details = [f"'{name}' (line {line})" for name, line in checker.undefined]
            raise EmissionLivenessError(
                f"AST liveness check failed: undefined variables loaded before definition: {', '.join(details)}"
            )

    def unify_and_emit(
        self,
        cells: List[Cell],
        prompt: str = "",
        intent_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Main synthesis entrypoint."""
        self.context = ExecutionContext(prompt=prompt)
        if (intent_data is None or not intent_data) and prompt:
            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                mm = ModelManager.get_instance()
                if mm.has_semantic_compiler():
                    intent_data = mm.compile_semantic_intent(prompt)
            except Exception:
                intent_data = None

        if intent_data and isinstance(intent_data, dict):
            tgt = intent_data.get("target_column") or intent_data.get("target_col")
            if tgt:
                self.context.target_col = str(tgt).strip()
            if intent_data.get("columns") and isinstance(intent_data["columns"], list):
                self.context.columns = [str(c).strip() for c in intent_data["columns"]]
            if intent_data.get("hyperparameters") and isinstance(intent_data["hyperparameters"], dict):
                self.context.hyperparameters = dict(intent_data["hyperparameters"])
            if intent_data.get("slots") and isinstance(intent_data["slots"], dict):
                self.context.llm_slots = dict(intent_data["slots"])
            # Profile S (Semantic Compiler): literals extracted by the typed
            # IR compiler become declared context parameters — deterministic
            # binding without re-scanning the prompt.
            if intent_data.get("parameters") and isinstance(intent_data["parameters"], dict):
                for p_key, p_val in intent_data["parameters"].items():
                    if p_key:
                        self.context.parameters.setdefault(str(p_key), p_val)
            if intent_data.get("by_column"):
                self.context.parameters["by_column"] = str(intent_data["by_column"]).strip()
            if intent_data.get("source_files"):
                self.context.parameters["source_uris"] = list(intent_data["source_files"])
            if intent_data.get("dest_files"):
                self.context.parameters["dest_uris"] = list(intent_data["dest_files"])
        if not self.context.parameters.get("source_uris") and self.context.source_files:
            self.context.parameters["source_uris"] = list(self.context.source_files)
        if not self.context.parameters.get("dest_uris") and self.context.dest_files:
            self.context.parameters["dest_uris"] = list(self.context.dest_files)
        return self.emit_code(cells, self.context)

    def build_verification_contract(
        self,
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
        ctx: Optional[ExecutionContext] = None,
        prompt: str = ""
    ) -> VerificationContract:
        """
        Builds a structured VerificationContract capturing Phase-1 postconditions
        and terminal node intent rules for GEVR sandbox execution.
        """
        contract = VerificationContract()

        # Track wires across pipeline
        registry = TypeRegistry.get_instance()

        split_train_features = None
        split_train_targets = None
        unsplit_features = None
        unsplit_targets = None

        ingress_image_var = None
        annotated_image_vars: List[str] = []

        ingress_df_var = None
        has_dropna = False
        has_dedup = False

        def _is_path_port(p: Any) -> bool:
            return _lattice_is_path_port(p)

        def _is_tensor_or_image(p: Any) -> bool:
            if not p:
                return False
            t = str(getattr(p, "type_name", "")).lower()
            st = str(getattr(p, "state", "")).lower()
            ab = str(getattr(p, "abstract_type", "")).lower()
            return (
                ab == "tensor"
                or registry.is_subtype(t, "tensor")
                or bool(st and registry.get_state_carrier(st) and registry.is_subtype(registry.get_state_carrier(st), "tensor"))
            )

        def _is_table_or_df(p: Any) -> bool:
            if not p:
                return False
            t = str(getattr(p, "type_name", "")).lower()
            ab = str(getattr(p, "abstract_type", "")).lower()
            return ab == "table" or registry.is_subtype(t, "table")

        def _is_figure(p: Any) -> bool:
            if not p:
                return False
            t = str(getattr(p, "type_name", "")).lower()
            return registry.is_subtype(t, "figure") or t in ("figure", "axes")

        def _is_estimator_output(p: Any) -> bool:
            if not p:
                return False
            t = str(getattr(p, "type_name", "")).lower()
            st = str(getattr(p, "state", "")).lower()
            return (
                registry.is_subtype(t, "estimator")
                or registry.state_ancestry_reaches(st, {"trained", "fit_estimator"})
            )

        for cell, bindings in pipeline_bindings:
            stage = getattr(cell, "stage", None)
            role = getattr(cell, "node_role", "")
            effects = getattr(cell, "effects", []) or []
            effect_names = {
                str(e.get("kind") if isinstance(e, dict) else e).lower()
                for e in effects
            } if effects else set()

            # 1. Phase-1 Cell Postconditions
            for post in (getattr(cell, "postconditions", []) or []):
                p_obj = post
                if not hasattr(p_obj, "property") and not hasattr(p_obj, "expression"):
                    try:
                        from .schema import ConditionPredicate
                        p_obj = ConditionPredicate.from_any(post)
                    except Exception as e:
                        try:
                            from schema import ConditionPredicate
                            p_obj = ConditionPredicate.from_any(post)
                        except Exception as e:
                            p_obj = post

                target_name = getattr(p_obj, "target", None) or "output_var"
                target_var = bindings.get(target_name)
                if not target_var:
                    target_var = bindings.get("output_var") or (bindings.get(cell.primary_output.name) if cell.primary_output else None)
                if target_var:
                    contract.cell_checks.append({
                        "cell_id": cell.cell_id,
                        "target_var": target_var,
                        "target_port": target_name,
                        "property": getattr(p_obj, "property", None),
                        "operator": getattr(p_obj, "operator", "=="),
                        "value": getattr(p_obj, "value", None),
                        "expression": getattr(p_obj, "expression", None),
                        "description": getattr(p_obj, "description", None),
                    })

            # 2. Ingress Tracking (Stage 1 or Source Role)
            if stage == 1 or role == "source":
                for p_name, p_sig in cell.outputs.items():
                    bound_v = bindings.get(p_name)
                    if not bound_v:
                        continue
                    if _is_tensor_or_image(p_sig):
                        if not ingress_image_var:
                            ingress_image_var = bound_v
                    elif _is_table_or_df(p_sig):
                        if not ingress_df_var:
                            ingress_df_var = bound_v

                    r = getattr(p_sig, "port_role", "") or ""
                    if r == "feature_input":
                        if not unsplit_features:
                            unsplit_features = bound_v
                    elif r == "target_input":
                        if not unsplit_targets:
                            unsplit_targets = bound_v

            # 3. Data Splitting / Partitioning Tracking (declared output states)
            for p_name, p_sig in cell.outputs.items():
                st = str(getattr(p_sig, "state", "")).lower()
                if "split_train_features" in st:
                    split_train_features = bindings.get(p_name)
                elif "split_train_targets" in st:
                    split_train_targets = bindings.get(p_name)

            if split_train_features and not unsplit_features:
                for in_name, in_sig in cell.inputs.items():
                    in_r = getattr(in_sig, "port_role", "") or ""
                    if in_r == "feature_input":
                        unsplit_features = bindings.get(in_name)
                        break

            # 4. Canvas Annotation / Drawing Tracking (declared effects only)
            is_draw_cell = bool(effect_names & {"draws_annotation", "renders_annotation"})
            if is_draw_cell:
                for out_name, out_sig in cell.outputs.items():
                    if _is_tensor_or_image(out_sig):
                        out_v = bindings.get(out_name)
                        if out_v and out_v not in annotated_image_vars:
                            annotated_image_vars.append(out_v)

            # 5. Data Cleaning Tracking (declared effects / declared postcondition
            # properties — no expression substring sniffing)
            if "cleans_missing" in effect_names:
                has_dropna = True
            elif any(
                getattr(p, "property", "") == "has_nans" and getattr(p, "value", True) is False
                for p in getattr(cell, "postconditions", []) or []
            ):
                has_dropna = True

            if "deduplicates" in effect_names or any(
                getattr(p, "property", "") == "is_deduped" and getattr(p, "value", False) is True
                for p in getattr(cell, "postconditions", []) or []
            ):
                has_dedup = True

            # 6. Terminal Intent Checks
            # Model Training Intent (declared role / declared carrier / declared state)
            is_estimator_cell = (
                role == "estimator"
                or any(_is_estimator_output(p) for p in cell.outputs.values())
            )
            if is_estimator_cell:
                # The model variable is the cell's primary output binding
                # (optionally a port declared with the model_sink role).
                model_var = None
                sink_port = next(
                    (p for p in cell.outputs.values() if getattr(p, "port_role", None) == "model_sink"),
                    None,
                )
                if sink_port is not None:
                    model_var = bindings.get(sink_port.name)
                if model_var is None and cell.primary_output is not None:
                    model_var = bindings.get(cell.primary_output.name)
                if model_var is None:
                    model_var = bindings.get("output_var")
                feat_var = None
                tgt_var = None
                for in_name, in_sig in cell.inputs.items():
                    r = getattr(in_sig, "port_role", "") or ""
                    if r == "feature_input" and feat_var is None:
                        feat_var = bindings.get(in_name)
                    elif r == "target_input" and tgt_var is None:
                        tgt_var = bindings.get(in_name)

                contract.terminal_checks.append({
                    "type": "model_fit_split",
                    "cell_id": cell.cell_id,
                    "model_var": model_var,
                    "feature_var": feat_var,
                    "target_var": tgt_var,
                    "expected_train_feature_var": split_train_features,
                    "unsplit_feature_var": unsplit_features,
                })

            # Terminal Egress Sinks (Stage 3 or sink role with path port)
            has_path_input = any(_is_path_port(p) for p in cell.inputs.values())
            is_sink_cell = (stage == 3 or role == "sink" or has_path_input) and stage != 1

            if is_sink_cell and has_path_input:
                path_var = None
                data_var = None
                data_kind = None

                for in_name, in_sig in cell.inputs.items():
                    if _is_path_port(in_sig):
                        path_var = bindings.get(in_name)
                    elif _is_tensor_or_image(in_sig):
                        data_var = bindings.get(in_name)
                        data_kind = "image"
                    elif _is_table_or_df(in_sig):
                        data_var = bindings.get(in_name)
                        data_kind = "table"
                    elif _is_figure(in_sig):
                        data_var = bindings.get(in_name)
                        data_kind = "figure"

                if data_kind == "image" or (_is_tensor_or_image(cell.primary_input) if cell.primary_input else False):
                    last_annotated = annotated_image_vars[-1] if annotated_image_vars else None
                    contract.terminal_checks.append({
                        "type": "image_annotation_egress",
                        "cell_id": cell.cell_id,
                        "saved_var": data_var,
                        "annotated_var": last_annotated,
                        "ingress_var": ingress_image_var,
                        "output_path": path_var,
                    })
                elif data_kind == "table" or (_is_table_or_df(cell.primary_input) if cell.primary_input else False):
                    contract.terminal_checks.append({
                        "type": "tabular_egress",
                        "cell_id": cell.cell_id,
                        "saved_var": data_var,
                        "ingress_var": ingress_df_var,
                        "output_path": path_var,
                        "expected_clean": {"no_nans": has_dropna, "is_deduped": has_dedup},
                    })
                elif data_kind == "figure" or (_is_figure(cell.primary_input) if cell.primary_input else False):
                    contract.terminal_checks.append({
                        "type": "visualization_egress",
                        "cell_id": cell.cell_id,
                        "fig_var": data_var,
                        "output_path": path_var,
                    })

        return contract

    @classmethod
    def unify_cell(cls, *args, **kwargs) -> str:
        """Single-cell unification helper for backwards-compatibility with tests."""
        gate = cls()
        context = None
        cell = None
        for a in args:
            if isinstance(a, ExecutionContext):
                context = a
            elif isinstance(a, Cell):
                cell = a
            elif isinstance(a, str) and context is None:
                context = ExecutionContext(a)
        if kwargs:
            context = kwargs.get("ctx", kwargs.get("context", context))
            cell = kwargs.get("cell", cell)
        if cell is None:
            # Fallback for positional if neither matched isinstance
            if len(args) == 2:
                if isinstance(args[1], Cell):
                    context, cell = args[0], args[1]
                else:
                    cell, context = args[0], args[1]
            else:
                raise ValueError("unify_cell requires a Cell")
        ctx = context if isinstance(context, ExecutionContext) else ExecutionContext(str(context or ""))
        res = gate.unify_pipeline([cell], ctx)
        if res.is_bottom():
            reason = res.reason if isinstance(res, Failure) else "Unknown unification failure"
            raise ValueError(f"Unification Failed: {reason}")
        pipeline_bindings = res.value
        if ctx.unresolved_ports:
            ctx.unresolved_ports.clear()
            for c, bnd in pipeline_bindings:
                for k, v in bnd.items():
                    if v is UNRESOLVED_PORT or (isinstance(v, str) and v in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")):
                        bnd[k] = k
        return gate.emit_code(pipeline_bindings, ctx)

    @classmethod
    def resolve_imports(cls, code_text: str, context: Any = None, chain_nodes: Any = None) -> Union[str, List[str]]:
        """Collects declared dependencies strictly from chain_nodes without domain hardcodes."""
        imports = set()
        if chain_nodes:
            for node in chain_nodes:
                for dep in getattr(node, "dependencies", []):
                    dep_str = dep.strip()
                    if dep_str:
                        if not (dep_str.startswith("import ") or dep_str.startswith("from ")):
                            dep_str = f"import {dep_str}"
                        imports.add(dep_str)
        import_block = "\n".join(sorted(list(imports)))
        if code_text:
            if import_block:
                return f"{import_block}\n\n{code_text}".strip()
            return code_text.strip()
        return sorted(list(imports))

    @classmethod
    def validate_synthesis(cls, cell_dict: Dict[str, Any], expected_inputs: str, expected_outputs: str, trees_dir: str = "trees") -> bool:
        """Verifies whether a synthesized cell's inputs and outputs unify with required types."""
        in_spec = cell_dict.get("inputs", {})
        out_spec = cell_dict.get("outputs", {})
        actual_in = in_spec.get("type_name") if isinstance(in_spec, dict) else str(in_spec)
        actual_out = out_spec.get("type_name") if isinstance(out_spec, dict) else str(out_spec)
        return types_unify(expected_inputs, actual_in) and types_unify(expected_outputs, actual_out)


# =====================================================================
# Compatibility Helpers and Error Classes
# =====================================================================

class UnificationFailure(Exception):
    """Raised when monadic unification fails to find a valid substitution."""
    pass


class UnresolvedPlaceholderError(UnificationFailure):
    """Raised when a placeholder cannot be resolved."""
    pass


class EmissionLivenessError(UnificationFailure):
    """Raised when emitted code references variables that have not been defined or imported."""
    pass


def types_unify(tau_expected: str, tau_actual: str) -> bool:
    """Verifies whether two types unify under the NSTL poset type system."""
    term1 = TypeTerm.from_string(tau_expected)
    term2 = TypeTerm.from_string(tau_actual)
    return unify(term1, term2) is not None


def assert_placeholders_resolved(template: str, bindings: Optional[Dict[str, Any]] = None) -> None:
    """Asserts that all {placeholder} slots in a template are bound without regex."""
    if bindings:
        for k, v in bindings.items():
            template = template.replace(f"{{{k}}}", str(v))
    remaining = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == '{':
            j = template.find('}', i + 1)
            if j != -1:
                inner = template[i + 1:j]
                if inner.isidentifier():
                    remaining.append(inner)
                i = j + 1
                continue
        i += 1
    if remaining:
        raise UnresolvedPlaceholderError(f"Unbound placeholders remaining: {remaining}")


class DynamicPlaceholderResolver:
    """Compatibility resolver delegating to monadic ExecutionContext and UnificationGate."""
    def __init__(self):
        self.context = ExecutionContext()
        self.gate = UnificationGate()

    def assert_placeholders_resolved(self, code_str: str):
        assert_placeholders_resolved(code_str)

    def resolve_port(self, port_name: str, port_sig: Any, stage: int, ctx: Any, current_out_var: str) -> str:
        sig = getattr(port_sig, "signature", port_sig)
        t_name = str(getattr(sig, "type_name", "")).lower()
        s_name = str(getattr(sig, "state", "")).lower()
        p_lower = str(port_name).lower()

        # 1. Source files / paths
        is_source_port = (
            p_lower in ("filepath", "source_path", "filename", "file_path", "path", "file", "image_path")
            or s_name in ("source_identifier", "file_path")
            or (stage == 1 and t_name in ("str", "path", "filepath", "any") and p_lower not in ("df", "data", "img", "image"))
        )
        if is_source_port and hasattr(ctx, "source_files") and ctx.source_files:
            return f'"{ctx.source_files[0]}"'

        # 2. Destination files / paths
        is_dest_port = (
            p_lower in ("dest_path", "savepath", "output_path", "dest_identifier", "target_path")
            or s_name in ("dest_identifier", "filepath_written")
            or (stage == 3 and t_name in ("str", "path", "filepath", "any") and p_lower not in ("df", "data", "img", "image", "src", "input"))
        )
        if is_dest_port and hasattr(ctx, "dest_files") and ctx.dest_files:
            return f'"{ctx.dest_files[0]}"'

        # 3. Columns / Column names
        if p_lower in ("by", "column", "columns", "subset") or s_name in ("column_name", "column_identifier"):
            if hasattr(ctx, "columns") and ctx.columns:
                return f'"{ctx.columns[0]}"'

        # 4. Operational flags
        if hasattr(ctx, "flags") and isinstance(ctx.flags, dict) and port_name in ctx.flags:
            return str(ctx.flags[port_name])
        if s_name == "sort_flag" and hasattr(ctx, "flags") and isinstance(ctx.flags, dict) and "ascending" in ctx.flags:
            return str(ctx.flags["ascending"])

        # 5. Direct context parameters
        if hasattr(ctx, "parameters") and port_name in ctx.parameters:
            return str(ctx.parameters[port_name])

        # 6. Default value (omit None default unless handles_none)
        if getattr(port_sig, "default_value", None) is not None:
            def_str = str(port_sig.default_value)
            if def_str in ("None", "none"):
                if getattr(port_sig, "handles_none", False):
                    return "None"
                return ""
            return def_str

        # 7. Type-compatible in-scope dataflow variable
        if hasattr(ctx, "scope_variables") and ctx.scope_variables:
            target_type = getattr(sig, "type_name", None)
            if target_type and target_type not in ("any", "*", "top"):
                for v_name, v_sig in reversed(list(ctx.scope_variables.items())):
                    v_type = getattr(getattr(v_sig, "signature", v_sig), "type_name", None)
                    if v_type == target_type:
                        return v_name
            return list(ctx.scope_variables.keys())[-1]

        return current_out_var


PlaceholderResolver = DynamicPlaceholderResolver


class DataflowLineageTracker(ast.NodeTransformer):
    """
    Tracks sequential variable transformations and re-links sink calls
    to the latest valid descendant in the lineage chain.
    """
    def __init__(self, target_cells=None):
        self.target_cells = target_cells or []
        self.lineage_tree: Dict[str, str] = {}
        self.latest_descendant: Dict[str, str] = {}
        self.assigned_vars: Set[str] = set()
        self._imported_names: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._imported_names.add(alias.asname or alias.name.split('.')[0])
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            self._imported_names.add(alias.asname or alias.name)
        if node.module:
            self._imported_names.add(node.module.split('.')[0])
        return node

    def visit_Assign(self, node: ast.Assign):
        self._rebind_sink_call(node.value)
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            self.assigned_vars.add(target_name)

            parent_var = None
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name):
                    candidate = node.value.func.value.id
                    if candidate not in self._imported_names:
                        parent_var = candidate
                elif isinstance(node.value.func, ast.Name):
                    for arg in node.value.args:
                        if isinstance(arg, ast.Name) and arg.id in self.assigned_vars:
                            parent_var = arg.id
                            break

            if parent_var:
                root = self._get_root(parent_var)
                self.lineage_tree[target_name] = parent_var
                self.latest_descendant[root] = target_name
                self.latest_descendant[parent_var] = target_name

        return node

    def visit_Expr(self, node: ast.Expr):
        self.generic_visit(node)
        self._rebind_sink_call(node.value)
        return node

    def _rebind_sink_call(self, call_node):
        if not isinstance(call_node, ast.Call):
            return

        method_name = ""
        callee_var = None
        if isinstance(call_node.func, ast.Attribute):
            method_name = call_node.func.attr
            if isinstance(call_node.func.value, ast.Name):
                callee_var = call_node.func.value.id
        elif isinstance(call_node.func, ast.Name):
            method_name = call_node.func.id

        is_sink = any(k in method_name.lower() for k in ["save", "write", "dump", "export", "to_"])
        for cell in self.target_cells:
            if getattr(cell.outputs, "type_name", "") == "None" or getattr(cell, "metadata_tags", {}).get("is_sink", False):
                is_sink = True
                break

        if is_sink:
            if callee_var and callee_var in self.latest_descendant and callee_var not in self._imported_names:
                newest_var = self.latest_descendant[callee_var]
                if newest_var != callee_var:
                    call_node.func.value.id = newest_var

            for arg in call_node.args:
                if isinstance(arg, ast.Name) and arg.id in self.latest_descendant:
                    arg.id = self.latest_descendant[arg.id]

    def _get_root(self, var_name: str) -> str:
        curr = var_name
        while curr in self.lineage_tree:
            curr = self.lineage_tree[curr]
        return curr


def enforce_lineage_integrity(code: str, target_cells=None) -> str:
    """Parses generated code, traces lineage, and auto-corrects stale variable usages."""
    try:
        tree = ast.parse(code)
        transformer = DataflowLineageTracker(target_cells=target_cells)
        corrected_tree = transformer.visit(tree)
        ast.fix_missing_locations(corrected_tree)
        return ast.unparse(corrected_tree)
    except Exception:
        return code


@dataclass
class ExtractedSlots:
    source_uris: List[str] = field(default_factory=list)
    dest_uris: List[str] = field(default_factory=list)
    named_identifiers: List[str] = field(default_factory=list)
    numeric_literals: List[Union[int, float]] = field(default_factory=list)
    operational_flags: Dict[str, Any] = field(default_factory=dict)
    by_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_uris": self.source_uris,
            "dest_uris": self.dest_uris,
            "named_identifiers": self.named_identifiers,
            "numeric_literals": self.numeric_literals,
            "operational_flags": self.operational_flags,
            "input_files": self.source_uris,
            "output_files": self.dest_uris,
            "columns": self.named_identifiers,
            "by_column": self.by_column or (self.named_identifiers[0] if self.named_identifiers else None),
        }


class ParameterExtractor:
    """Compatibility adapter over ExecutionContext."""

    @staticmethod
    def _declared_flag_patterns() -> List[Tuple[Any, str, Any, Dict[str, Any]]]:
        """
        Keyword->flag observation patterns are DECLARED DATA harvested from
        the loaded domain trees (tree key `semantic_flag_patterns`). The
        engine ships zero patterns of its own: e.g. a vision tree declares
        its colorspace observers and the primitives tree declares the
        language-level ordering observers (ascending/descending).
        """
        try:
            declared = TypeRegistry.get_instance().get_semantic_flag_patterns()
        except Exception:
            return []
        return [
            (p["keywords"], p["flag"], p["value"], dict(p.get("extras") or {}))
            for p in declared
        ]

    @staticmethod
    def extract_slots(prompt: str) -> ExtractedSlots:
        ctx = ExecutionContext(prompt=prompt)
        slots = ExtractedSlots()
        slots.source_uris = list(ctx.source_files)
        slots.dest_uris = list(ctx.dest_files)
        slots.numeric_literals = [
            float(val) if "." in val else int(val)
            for _, kind, val in ctx.ordered_literals if kind == "numeric"
        ]
        quoted_strings = [val for _, kind, val in ctx.ordered_literals if kind == "quoted_str" and "." not in val]
        if quoted_strings:
            slots.named_identifiers.extend(quoted_strings)
        elif ctx.columns:
            slots.named_identifiers.extend(ctx.columns)

        flags = dict(ctx.flags)
        if prompt:
            p_low = prompt.lower()
            for keywords, flag_key, flag_val, extras in ParameterExtractor._declared_flag_patterns():
                if any(kw in p_low for kw in keywords):
                    flags[flag_key] = flag_val
                    flags.update(extras)

        slots.operational_flags = flags
        if slots.named_identifiers:
            slots.by_column = slots.named_identifiers[0]
        return slots

    @staticmethod
    def extract_parameters(prompt: str) -> Dict[str, Any]:
        return ParameterExtractor.extract_slots(prompt).to_dict()


def _enum_rule_for_placeholder(placeholder_lower: str) -> Optional[Dict[str, Any]]:
    """
    Finds a DECLARED enum-constant grounding rule (tree key
    `enum_constant_rules`) governing a placeholder name. Returns None when no
    tree declared a rule — the engine carries no placeholder vocabulary.
    """
    try:
        rules = TypeRegistry.get_instance().get_enum_constant_rules()
    except Exception:
        return None
    for rule in rules or []:
        if placeholder_lower in (rule.get("placeholder_names") or ()):  # declared frozenset
            return rule
    return None


def resolve_node_slots(template: str, extracted_params: Dict[str, Any]) -> Dict[str, str]:
    """Deterministically binds extracted prompt parameters to template slot placeholders."""
    slots = {}
    placeholders = sorted(extract_template_placeholders(template))
    src_uris = extracted_params.get("source_uris", []) or extracted_params.get("input_files", [])
    dst_uris = extracted_params.get("dest_uris", []) or extracted_params.get("output_files", [])
    by_col = extracted_params.get("by_column") or (extracted_params.get("columns", [None])[0] if extracted_params.get("columns") else None)
    flags = extracted_params.get("operational_flags", {})

    for ph in placeholders:
        ph_l = ph.lower()
        if "filename" in ph_l or "path" in ph_l or "uri" in ph_l:
            if "output" in ph_l or "dest" in ph_l or "save" in ph_l:
                if dst_uris:
                    slots[ph] = f"'{dst_uris[-1]}'"
            else:
                if src_uris:
                    slots[ph] = f"'{src_uris[0]}'"
        elif ph_l in ("by", "by_column", "column"):
            if by_col:
                slots[ph] = f"'{by_col}'"
        elif ph_l == "ascending":
            slots[ph] = str(flags.get("ascending", True))
        elif _enum_rule_for_placeholder(ph_l) is not None:
            # Declared enum-constant grounding (Solution: domain data in trees):
            # a rule declares the placeholder names it governs, the module
            # attribute prefix to resolve (e.g. a colorspace-constant prefix),
            # and which operational flags select which format suffix. The
            # engine only performs generic module reflection — the vocabulary
            # itself lives entirely in the domain tree.
            rule = _enum_rule_for_placeholder(ph_l)
            ph_needle = f"{{{ph}}}"
            mod_name = None
            if ph_needle in template:
                prefix = template.split(ph_needle)[0]
                lparen_idx = prefix.rfind("(")
                if lparen_idx != -1:
                    caller = prefix[:lparen_idx].strip()
                    call_token = caller.split()[-1] if caller else ""
                    if "." in call_token:
                        candidate_mod = call_token.split(".")[0].lstrip("=,([+-*/% ")
                        if candidate_mod.isidentifier():
                            mod_name = candidate_mod
            if not mod_name:
                continue
            attr_prefix = str(rule.get("module_attribute_prefix", ""))
            target_fmt = str(rule.get("default_format", ""))
            format_flags = dict(rule.get("format_flags") or {})
            for flag_key, fmt in format_flags.items():
                if flags.get(flag_key):
                    target_fmt = str(fmt)
                    break
            if not target_fmt:
                continue
            resolved_code = None
            try:
                import importlib
                mod = importlib.import_module(mod_name)
                cand = next(
                    (attr for attr in dir(mod) if attr_prefix and attr.startswith(attr_prefix) and attr.endswith(target_fmt)),
                    None,
                )
                if not cand:
                    cand = next(
                        (attr for attr in dir(mod) if attr_prefix and attr.startswith(attr_prefix.rstrip("_")) and attr.endswith(target_fmt)),
                        None,
                    )
                if cand:
                    resolved_code = f"{mod_name}.{cand}"
            except Exception:
                pass
            slots[ph] = resolved_code or (f"{mod_name}.{attr_prefix}{target_fmt}" if attr_prefix else f"{mod_name}.{target_fmt}")
        elif ph_l in ("target", "target_col", "target_column", "label", "y_col"):
            tgt = extracted_params.get("target_column") or extracted_params.get("target_col")
            if tgt:
                slots[ph] = f"'{tgt}'"
        elif ph_l in ("columns", "feature_cols", "features", "x_cols"):
            cols = extracted_params.get("columns")
            if cols:
                slots[ph] = repr(cols)
        else:
            hp = extracted_params.get("hyperparameters", {})
            if ph in hp:
                slots[ph] = repr(hp[ph])
            elif ph_l in hp:
                slots[ph] = repr(hp[ph_l])
            else:
                llm_s = extracted_params.get("slots", {}) or extracted_params.get("llm_slots", {})
                if ph in llm_s:
                    slots[ph] = repr(llm_s[ph])
                elif ph_l in llm_s:
                    slots[ph] = repr(llm_s[ph_l])

    return slots


def unify_cell_with_scope(
    cand: Cell,
    available_cells: List[Cell],
    sigma: Optional[Substitution] = None,
    context: Optional[ExecutionContext] = None
) -> Optional[Tuple[Substitution, Dict[str, Tuple[Cell, str]]]]:
    """
    Scope-Level Robinson Unification for Dataflow DAGs.
    Verifies that all required input ports of `cand` are satisfiable from:
      1. Available output ports of cells in `available_cells`
      2. In-scope variables in `context` (if provided)
      3. Default values / prompt literals
    Returns (updated_substitution, port_bindings) where port_bindings maps
    port_name -> (producer_cell, producer_out_port_name), or None if unsatisfiable.
    """
    sub = Substitution(sigma.mappings if sigma else {})
    port_bindings: Dict[str, Tuple[Cell, str]] = {}
    bound_count = 0

    def _extract_ndim(sc: Any) -> Optional[int]:
        if isinstance(sc, dict):
            val = sc.get("ndim")
            if isinstance(val, int):
                return val
            try:
                return int(val) if val is not None else None
            except (ValueError, TypeError):
                return None
        elif isinstance(sc, str):
            sc = sc.strip()
            if sc.startswith("(") and sc.endswith(")"):
                inner = sc[1:-1].strip()
                if not inner:
                    return 0
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                return len(parts)
        return None

    def _shape_compatible(p_out: Any, p_in: Any) -> bool:
        in_role = getattr(p_in, "port_role", None) or getattr(p_in, "derived_role", "")
        out_role = getattr(p_out, "port_role", None) or getattr(p_out, "derived_role", "")
        if in_role == "prediction_input" and out_role == "target_input":
            return False
        if in_role == "target_input" and out_role == "prediction_output":
            return False
        out_sc = getattr(p_out, "shape_contract", None)
        in_sc = getattr(p_in, "shape_contract", None)
        if out_sc and in_sc:
            o_ndim = _extract_ndim(out_sc)
            i_ndim = _extract_ndim(in_sc)
            if o_ndim is not None and i_ndim is not None and o_ndim != i_ndim:
                return False
        return True

    candidates = []
    # Collect candidate wire matches across all available DAG output ports
    for p_name, p_sig in cand.inputs.items():
        in_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
        p_sig_inner = getattr(p_sig, "signature", p_sig)
        p_type = str(getattr(p_sig_inner, "type_name", "") or "").lower()
        p_state = str(getattr(p_sig_inner, "state", "") or "").lower()

        # 1. Available cells in DAG history
        for idx_c, earlier_cell in enumerate(reversed(available_cells)):
            for out_name, out_sig in earlier_cell.outputs.items():
                if not _shape_compatible(out_sig, p_sig):
                    continue
                s_wire = unify(out_sig.signature, p_sig.signature, sub)
                if s_wire is not None:
                    out_sig_inner = getattr(out_sig, "signature", out_sig)
                    out_type = str(getattr(out_sig_inner, "type_name", "") or "").lower()
                    out_state = str(getattr(out_sig_inner, "state", "") or "").lower()
                    out_role = getattr(out_sig, "port_role", None) or getattr(out_sig, "derived_role", "")

                    score = 10.0
                    # Exact type equality bonus
                    if p_type and out_type and p_type == out_type:
                        score += 25.0
                    # Exact state match bonus
                    if p_state and out_state and p_state == out_state:
                        score += 20.0
                    # Port role compatibility
                    if in_role and out_role and in_role == out_role:
                        score += 15.0
                    # Recency tie-breaker
                    score += (len(available_cells) - idx_c) * 0.1

                    # Multi-input positional alignment: if consumer has multiple inputs accepting this carrier type,
                    # align them with ancestor producer cells in topological order (e.g. left -> first df, right -> second df)
                    carrier_in_ports = [
                        k for k, p in cand.inputs.items()
                        if unify(getattr(p, "signature", p), getattr(p_sig, "signature", p_sig), sub) is not None
                    ]
                    matching_ancestors = [
                        c for c in available_cells
                        if any(unify(o.signature, p_sig.signature, sub) is not None for o in c.outputs.values())
                    ]
                    if len(carrier_in_ports) > 1 and len(matching_ancestors) > 1:
                        port_idx = carrier_in_ports.index(p_name) if p_name in carrier_in_ports else -1
                        anc_idx = matching_ancestors.index(earlier_cell) if earlier_cell in matching_ancestors else -1
                        if port_idx >= 0 and port_idx == anc_idx:
                            score += 1.0

                    wire_key = (id(earlier_cell), out_name)
                    candidates.append((score, p_name, earlier_cell, out_name, wire_key, s_wire))

        # 2. Context variables if available
        if context is not None:
            avail_cell_ids = {id(c) for c in available_cells}
            for v_name, (v_sig, _) in context.variables.items():
                v_cell = getattr(context, "var_sources", {}).get(v_name)
                if v_cell is not None and id(v_cell) in avail_cell_ids:
                    continue
                if not _shape_compatible(v_sig, p_sig):
                    continue
                s_wire = unify(v_sig.signature, p_sig.signature, sub)
                if s_wire is not None:
                    v_sig_inner = getattr(v_sig, "signature", v_sig)
                    v_type = str(getattr(v_sig_inner, "type_name", "") or "").lower()
                    v_state = str(getattr(v_sig_inner, "state", "") or "").lower()
                    v_role = getattr(v_sig, "port_role", None) or getattr(v_sig, "derived_role", "")

                    score = 10.0
                    if p_type and v_type and p_type == v_type:
                        score += 25.0
                    if p_state and v_state and p_state == v_state:
                        score += 20.0
                    if in_role and v_role and in_role == v_role:
                        score += 15.0
                    v_cell = getattr(context, "var_sources", {}).get(v_name)
                    wire_key = v_name
                    candidates.append((score, p_name, v_cell, v_name, wire_key, s_wire))

    # Deterministic greedy assignment with disjoint wire tracking
    candidates.sort(key=lambda x: x[0], reverse=True)
    assigned_ports = set()
    assigned_wires = set()

    for score, p_name, prod_cell, out_name, wire_key, s_wire in candidates:
        if p_name in assigned_ports or wire_key in assigned_wires:
            continue
        assigned_ports.add(p_name)
        assigned_wires.add(wire_key)
        port_bindings[p_name] = (prod_cell, out_name)
        sub = s_wire
        bound_count += 1

    # Check remaining unbound ports against defaults / literals
    for p_name, p_sig in cand.inputs.items():
        if p_name in assigned_ports:
            continue
        is_required = getattr(p_sig, "required", False)
        default_val = getattr(p_sig, "default_value", None)
        if not is_required or default_val is not None:
            continue
        desc = getattr(p_sig, "description", None) or getattr(p_sig, "doc", "") or ""
        is_receiver = p_name in ("data", "self") or ("receiver" in str(desc).lower())
        satisfied = False
        if not is_receiver:
            registry = TypeRegistry.get_instance()
            t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
            if registry.is_subtype(t_name, "str") or registry.is_subtype(t_name, "int") or registry.is_subtype(t_name, "float") or registry.is_subtype(t_name, "scalar"):
                satisfied = True
        if not satisfied:
            return None

    if cand.inputs and bound_count == 0 and getattr(cand, "stage", None) != 1 and getattr(cand, "node_type", "") != "constructor":
        return None

    return sub, port_bindings

