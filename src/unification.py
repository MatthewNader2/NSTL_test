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
import re
import json
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable, Generic, TypeVar, FrozenSet

from log_config import get_logger

try:
    from .lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry
    from .tokenizer import CellTokenizer, normalize_token
except (ImportError, ValueError):
    from lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry
    from tokenizer import CellTokenizer, normalize_token

logger = get_logger('unification')

# Module-level tokenizer alias: resolve_literal_for_port contains local
# `from .tokenizer import CellTokenizer` shadowing inside optional branches,
# which would make the global name unavailable to unconditional code paths.
_TOKENIZER = CellTokenizer

T = TypeVar('T')
U = TypeVar('U')

# English sentence-connective function words (LANGUAGE-level primitives, not
# domain vocabulary): a capitalized occurrence of one of these mid-prompt is a
# sentence connective, never a referential identifier.
_SENTENCE_CONNECTIVES = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "from", "by", "with",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "them", "they", "their", "this", "that", "these", "those",
    "but", "if", "then", "when", "while", "into", "also", "plus", "using", "use",
    "not", "no", "do", "does", "did", "can", "could", "should", "would", "will",
    "sure", "some", "each", "all", "both", "make",
})


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


TOP_TYPE_SET = {"any", "Any", "*", "top", "⊤", "object", "Object"}


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
        if s_clean in TOP_TYPE_SET:
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
        # Single-letter generic identifiers (T, S, K, V, U, etc.)
        if len(s_clean) == 1 and s_clean.isupper():
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
    """Typestate compound term: tau = (type_name, state, qualifiers)."""
    type_name: str
    state: str = "any"
    qualifiers: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)

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
        return TypestateTerm(type_name=t_resolved, state=self.state, qualifiers=self.qualifiers)

    def __repr__(self) -> str:
        return f"{self.type_name}[{self.state}]"


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
        tokens = re.findall(r"\b[A-Za-z_]\w*\b", term)
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


_UNIFY_BASE_CACHE: Dict[Tuple[TypeTerm, TypeTerm], Optional[Substitution]] = {}

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
        if cache_key in _UNIFY_BASE_CACHE:
            cached = _UNIFY_BASE_CACHE[cache_key]
            return Substitution(cached.mappings) if cached is not None else None

    sub = Substitution(sigma.mappings if sigma else {})

    t1 = t1.apply_substitution(sub)
    t2 = t2.apply_substitution(sub)

    # 1. Identity or Top
    if t1 == t2 or isinstance(t1, TopType) or isinstance(t2, TopType):
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = sub
        return sub

    # 2. Variable binding (Robinson first-order unification with occurs check)
    if isinstance(t1, TypeVariable):
        if isinstance(t2, TypeVariable) and t1.var_name == t2.var_name:
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = sub
            return sub
        if occurs_check(t1.var_name, t2, sub):
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = None
            return None  # Occurs check failure -> bottom
        sub.bind(t1.var_name, t2)
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = sub
        return sub

    if isinstance(t2, TypeVariable):
        if occurs_check(t2.var_name, t1, sub):
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = None
            return None  # Occurs check failure -> bottom
        sub.bind(t2.var_name, t1)
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = sub
        return sub

    # 2.3. Coproduct / Union unification (canonical injection)
    if isinstance(t1, UnionTypeTerm):
        for alt in t1.terms:
            sub_alt = unify(alt, t2, sub)
            if sub_alt is not None:
                if is_ground_query:
                    _UNIFY_BASE_CACHE[cache_key] = sub_alt
                return sub_alt
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = None
        return None

    if isinstance(t2, UnionTypeTerm):
        for alt in t2.terms:
            sub_alt = unify(t1, alt, sub)
            if sub_alt is not None:
                if is_ground_query:
                    _UNIFY_BASE_CACHE[cache_key] = sub_alt
                return sub_alt
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = None
        return None

    # 2.5. Generic container unification
    if isinstance(t1, GenericTypeTerm) and isinstance(t2, GenericTypeTerm):
        registry = TypeRegistry.get_instance()
        c1 = t1.constructor.lower()
        c2 = t2.constructor.lower()
        compat = (c1 == c2) or registry.is_subtype(c1, c2) or registry.is_subtype(c2, c1)
        if compat and len(t1.args) == len(t2.args):
            for a1, a2 in zip(t1.args, t2.args):
                sub = unify(a1, a2, sub)
                if sub is None:
                    if is_ground_query:
                        _UNIFY_BASE_CACHE[cache_key] = None
                    return None
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = sub
            return sub
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = None
        return None

    if isinstance(t1, GenericTypeTerm) and (isinstance(t2, AtomicType) or isinstance(t2, TypestateTerm)):
        registry = TypeRegistry.get_instance()
        target_name = t2.type_name if isinstance(t2, TypestateTerm) else t2.name
        if registry.is_subtype(t1.constructor, target_name):
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = sub
            return sub

    if isinstance(t2, GenericTypeTerm) and (isinstance(t1, AtomicType) or isinstance(t1, TypestateTerm)):
        registry = TypeRegistry.get_instance()
        source_name = t1.type_name if isinstance(t1, TypestateTerm) else t1.name
        if registry.is_subtype(source_name, t2.constructor):
            for arg in t2.args:
                if isinstance(arg, TypeVariable) and arg.var_name not in sub.mappings:
                    sub.bind(arg.var_name, TOP)
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = sub
            return sub

    # 3. Typestate term unification
    if isinstance(t1, TypestateTerm) and isinstance(t2, TypestateTerm):
        # State unification: if both specify a concrete state, they must match
        if t1.state != "any" and t2.state != "any":
            if t1.state.lower() != t2.state.lower():
                if is_ground_query:
                    _UNIFY_BASE_CACHE[cache_key] = None
                return None  # State mismatch -> bottom

        # Qualifier subset check
        if t2.qualifiers and not t2.qualifiers.issubset(t1.qualifiers):
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = None
            return None

        # Check if type_names contain generic definitions
        if ("[" in t1.type_name) or ("[" in t2.type_name) or (len(t1.type_name) == 1 and t1.type_name.isupper()) or (len(t2.type_name) == 1 and t2.type_name.isupper()):
            inner1 = TypeTerm.from_string(t1.type_name)
            inner2 = TypeTerm.from_string(t2.type_name)
            return unify(inner1, inner2, sub)

        # Type poset subtyping check: t1.type_name <= t2.type_name
        registry = TypeRegistry.get_instance()
        if not registry.is_subtype(t1.type_name, t2.type_name):
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = None
            return None  # Type mismatch -> bottom

        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = sub
        return sub

    # 4. Atomic type unification
    if isinstance(t1, AtomicType) and isinstance(t2, AtomicType):
        registry = TypeRegistry.get_instance()
        if registry.is_subtype(t1.name, t2.name):
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = sub
            return sub
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = None
        return None

    # 5. Mixed Atomic and Typestate unification
    if isinstance(t1, AtomicType) and isinstance(t2, TypestateTerm):
        registry = TypeRegistry.get_instance()
        if registry.is_subtype(t1.name, t2.type_name):
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = sub
            return sub
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = None
        return None

    if isinstance(t1, TypestateTerm) and isinstance(t2, AtomicType):
        registry = TypeRegistry.get_instance()
        if registry.is_subtype(t1.type_name, t2.name):
            if is_ground_query:
                _UNIFY_BASE_CACHE[cache_key] = sub
            return sub
        if is_ground_query:
            _UNIFY_BASE_CACHE[cache_key] = None
        return None

    if is_ground_query:
        _UNIFY_BASE_CACHE[cache_key] = None
    return None


def _to_type_term(item: Any) -> TypeTerm:
    if isinstance(item, TypeTerm):
        return item
    if isinstance(item, AlgebraicSignature):
        cached = getattr(item, "_cached_term", None)
        if cached is not None:
            return cached
        if item.is_top():
            res = TOP
        else:
            t_clean = item.type_name.strip()
            if t_clean.startswith("?") or "|" in t_clean or ("[" in t_clean and t_clean.endswith("]")) or (len(t_clean) == 1 and t_clean.isupper()):
                res = TypeTerm.from_string(t_clean)
            else:
                registry = TypeRegistry.get_instance()
                canonical = registry.canonical_name(t_clean)
                if canonical.lower() in ("any", "*", "top", "object", "unknown"):
                    res = TOP
                else:
                    res = TypestateTerm(type_name=canonical, state=item.state, qualifiers=item.qualifiers)
        try:
            item._cached_term = res
        except (AttributeError, TypeError):
            pass
        return res
    if isinstance(item, PortSignature):
        return _to_type_term(item.signature)
    registry = TypeRegistry.get_instance()
    if isinstance(item, str):
        canonical = registry.canonical_name(item)
        if canonical.lower() in ("any", "*", "top", "object", "unknown"):
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

    if isinstance(target, TypeTerm):
        return target.apply_substitution(sub)

    if isinstance(target, PortSignature):
        new_sig = substitute_generics(target.signature, sub)
        return PortSignature(
            name=target.name,
            signature=new_sig,
            required=target.required,
            default_value=target.default_value,
            doc=target.doc,
            domain=target.domain
        )

    if isinstance(target, AlgebraicSignature):
        new_type = substitute_generics(target.type_name, sub)
        return AlgebraicSignature(
            type_name=str(new_type),
            state=target.state,
            qualifiers=target.qualifiers
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


# =====================================================================
# 4. Domain-Agnostic Execution Context
# =====================================================================

class ExecutionContext:
    """
    Runtime execution scope holding bound variables and literal arguments.
    Operates strictly via algebraic typestates and substitutions with ZERO domain hardcodes.
    """
    def __init__(self, prompt: str = "", scope: Optional[Dict[str, Any]] = None):
        self._prompt = prompt or ""
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
        self.ordered_literals: List[Tuple[int, str, str]] = self._extract_universal_literals(self._prompt)
        # Bare-identifier role map: char position -> stemmed role context tokens
        # (e.g. pos(X) -> {"column"}). Drives role-conditioned identifier binding
        # and for-each multiplicity detection.
        self.identifier_roles: Dict[int, FrozenSet[str]] = self._build_identifier_role_map(self._prompt)
        if scope:
            for k, v in scope.items():
                self.declare_variable(k, v, k)

    @property
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, val: str):
        self._prompt = val or ""
        self.used_indices = set()
        self.consumed_tokens = set()
        self.consumed_members = set()
        self.ordered_literals = self._extract_universal_literals(self._prompt)
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
                    spans.append((i, "quoted_str", val))
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
            return any(s - 1 <= pos and pos + length <= s + len(v) + 2 for s, t, v in spans if t == "quoted_str")

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

            # Numeric tokens
            try:
                float(w)
                spans.append((pos, "numeric", w))
                continue
            except ValueError:
                pass

            # Bare referential identifiers: the way humans name columns, fields
            # and variables in prose WITHOUT quoting them ("normalize X column").
            # Structural orthography only: a short, capitalized, alphanumeric
            # token that no other literal kind claims. Sentence-initial words
            # and sentence connectives are excluded (a capitalized "The" mid-
            # prompt is a connective, not a name); this is language-level
            # orthography, not domain vocabulary.
            if (
                pos not in sentence_initial
                and ExecutionContext._is_bare_identifier_token(w)
            ):
                spans.append((pos, "identifier", w))
                continue

        # 3. Predicate / filter comparison expressions (e.g. "value > 100", "score <= 50", "x == 1")
        cmp_matches = re.finditer(r'\b([a-zA-Z_]\w*\s*(?:>|<|==|!=|>=|<=)\s*(?:\d+(?:\.\d+)?|[\'"][^\'"]+[\'"]))\b', prompt)
        for m in cmp_matches:
            spans.append((m.start(), "expr", m.group(1).strip()))

        # Order strictly by character position in prompt
        spans.sort(key=lambda x: x[0])
        return spans

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
        for w_idx, (pos, raw) in enumerate(words):
            w = raw.rstrip(".,;:)")
            if w:
                cleaned.append((pos, w))

        # Identifier candidates: interior words (never the first word of the
        # prompt — sentence-initial capitalization is grammatical, not naming)
        idents: List[Tuple[int, int, str]] = []  # (clean_idx, pos, token)
        for c_idx, (pos, w) in enumerate(cleaned):
            if c_idx == 0:
                continue
            if any(s <= pos <= e for s, e in quoted_ranges):
                continue
            if "." in w or "/" in w:
                continue
            try:
                float(w)
                continue
            except ValueError:
                pass
            if ExecutionContext._is_bare_identifier_token(w):
                idents.append((c_idx, pos, w))

        if not idents:
            return []

        def _stem_ctx(word: str) -> str:
            wl = word.lower()
            if wl in _SENTENCE_CONNECTIVES or len(wl) < 2:
                return ""
            st = normalize_token(wl)
            return st if len(st) >= 2 else ""

        # Role assignment: prefer the FOLLOWING context word ("X column"),
        # else the PRECEDING one ("columns X"). Identifiers sharing the same
        # role stem join one group; roleless identifiers form singleton groups
        # (they can still bind, but never drive multiplicity).
        groups: Dict[str, List[Tuple[int, str]]] = {}
        order: List[str] = []
        for c_idx, pos, w in idents:
            role = ""
            if c_idx + 1 < len(cleaned):
                role = _stem_ctx(cleaned[c_idx + 1][1])
            if not role and c_idx > 0:
                role = _stem_ctx(cleaned[c_idx - 1][1])
            key = role
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((pos, w))

        return [
            IdentifierGroup(members=members, role_tokens=frozenset({k}) if k else frozenset())
            for k in order
            for members in [groups[k]]
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
        if isinstance(port_sig, AlgebraicSignature):
            port_sig = PortSignature(name=name, signature=port_sig)
        elif not isinstance(port_sig, PortSignature):
            port_sig = PortSignature(name=name, signature=AlgebraicSignature(str(port_sig), "any"))
        self.variables[name] = (port_sig, expr or name)
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

        # Candidate word tokens from prompt, excluding self-referential collisions
        p_stem = normalize_token((param_name or "").lower()) if param_name else ""
        words = []
        for w in self.prompt.strip().split():
            clean_w = w.strip(" '\".,;:()[]{}=:")
            if len(clean_w) < 2 or clean_w.lower() in self.consumed_tokens:
                continue
            if p_stem:
                w_stem = normalize_token(clean_w.lower())
                if (
                    w_stem == p_stem
                    or clean_w.lower().startswith(p_stem)
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

        # Fallback: if param_name explicitly matches a prompt word, take next token
        p_name_lower = target_label.lower()
        words_lower = [w.lower() for w in words]
        if p_name_lower in words_lower:
            idx = words_lower.index(p_name_lower)
            if idx + 1 < len(words):
                candidate = words[idx + 1]
                self.consumed_tokens.add(candidate.lower())
                return candidate

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
        except Exception:
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
        except Exception:
            pass

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
        p_name = port_sig.name.lower()
        t_name = port_sig.type_name.lower()

        def _cell_has_path_port() -> bool:
            if not cell_inputs:
                return False
            for p in cell_inputs.values():
                tn = str(getattr(p, "type_name", "")).lower()
                if registry.is_subtype(tn, "filepath") or registry.is_subtype(tn, "path") or registry.is_subtype(tn, "uri"):
                    return True
            return False

        registry = TypeRegistry.get_instance()
        is_bool = registry.is_subtype(t_name, "bool")
        is_num = registry.is_subtype(t_name, "numeric") and not is_bool
        is_str = registry.is_subtype(t_name, "str") or t_name in ("str", "any")

        # 1. Parameter explicitly supplied in context
        if port_sig.name in self.parameters:
            val = self.parameters[port_sig.name]
            return json.dumps(val) if isinstance(val, str) else str(val)

        # 2. Dynamic Enum / Flag Constant Grounding via Domain Reflection & Vector Similarity
        if (t_name == "enum" or getattr(port_sig, "domain", None)) and self.prompt:
            domain_spec = getattr(port_sig, "domain", "") or ""
            resolved_enum = self._resolve_enum_constant(domain_spec, port_sig.state)
            if resolved_enum is not None:
                return resolved_enum

        # 3. Vector Polarity Projection for Boolean / Valuation Ports
        if is_bool and self.prompt:
            qualifier_map = dict(getattr(port_sig.signature, "qualifiers", []))
            pos_label = qualifier_map.get("positive", port_sig.name)
            neg_label = qualifier_map.get("negative", f"not {port_sig.name}")

            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                try:
                    from .tokenizer import CellTokenizer
                except (ImportError, ValueError):
                    from tokenizer import CellTokenizer
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

            if port_sig.required:
                if port_sig.default_value is not None:
                    return str(port_sig.default_value)
                return None
            return None

        # 4. Numeric literals for numeric ports
        if is_num:
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind == "numeric":
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
        if _port_state in ("expr", "condition", "filter_condition") or (
            _is_str_port and cell_stage == 2
        ):
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind in ("expr", "quoted_str"):
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 5. Stage 1 and Stage 3 Morphisms: Environmental Asset Grounding
        # file_asset literals flow only into path-typed ports. Plain str ports may
        # receive assets only when the cell declares NO dedicated path port, so that
        # auxiliary string parameters can never steal file assets from the sink/source.
        is_path_port = registry.is_subtype(t_name, "filepath") or registry.is_subtype(t_name, "path") or registry.is_subtype(t_name, "uri")
        if is_path_port or (cell_stage in (1, 3) and is_str and not _cell_has_path_port()):
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind in ("file_asset", "quoted_str"):
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 6. Stage 2 Morphism: Operational Parameter Extraction
        if (cell_stage == 2 or cell_stage is None) and is_str:
            # A. Check for unconsumed quoted string argument in prompt (e.g. 'age', 'cup')
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind == "quoted_str":
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 6b. Referential identifier grounding (role-conditioned).
        # Bare identifiers (X, Y, Z — the way humans name columns/fields in
        # prose) are a VALUE CLASS, bound under evidence, never by the port's
        # own name: the identifier's shared role context (e.g. "column") must
        # intersect the consuming cell's DECLARED identity vocabulary or the
        # port's DECLARED typestate role. This separates "what the slot is
        # called" from "what kind of thing goes in it".
        if (cell_stage == 2 or cell_stage is None) and port_sig.required and port_sig.default_value is None:
            _tn = t_name
            _is_strict_str = registry.is_subtype(_tn, "str") and _tn not in ("any", "*", "top", "")
            _state_tokens: Set[str] = set()
            _raw_state = str(getattr(port_sig, "state", "") or "")
            if _raw_state.lower() not in ("any", "default", ""):
                _state_tokens = _TOKENIZER.tokenize_identifier(_raw_state)
            if _is_strict_str or _state_tokens:
                _identity_scope: Set[str] = set(cell_tokens or set()) | _state_tokens
                for idx, (_, kind, val) in enumerate(self.ordered_literals):
                    if idx in self.used_indices or kind != "identifier":
                        continue
                    role_tokens = self.identifier_roles.get(self.ordered_literals[idx][0], frozenset())
                    if not role_tokens or not (role_tokens & _identity_scope):
                        continue
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 7. Port default value declared in tree schema
        if port_sig.default_value is not None:
            def_str = str(port_sig.default_value).strip()
            if def_str in ("True", "False", "None"):
                return def_str
            try:
                parsed = ast.parse(def_str, mode="eval").body
                if not is_str or not (isinstance(parsed, ast.Constant) and isinstance(parsed.value, str)):
                    return def_str
            except Exception:
                if not is_str:
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

class UnificationGate:
    """
    Formal Unification Gate verifying dataflow composition and emitting code.
    Contains ZERO hardcoded domain libraries or prompt-sniffing regexes.
    """
    def __init__(self):
        self.context = ExecutionContext()
        self.last_egress_paths: List[str] = []

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
            writers with a destination port (e.g. DataFrame writers) are Stage 2
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

        # 1. Primary input direct unification
        new_sigma = unify(out_sig.signature, consumer.primary_input.signature, current_sigma)
        if new_sigma is not None:
            return Success(new_sigma, new_sigma)

        # 2. Multi-Port Monoidal Matching: check if producer output unifies with ANY input of consumer
        for p_name, p_port in consumer.inputs.items():
            new_sigma = unify(out_sig.signature, p_port.signature, current_sigma)
            if new_sigma is not None:
                return Success(new_sigma, new_sigma)

        # 3. Port-sharing delta check: if consumer's primary input is satisfiable by in-scope variable
        if context is not None:
            for v_name, (v_sig, _) in context.variables.items():
                v_u = unify(v_sig.signature, consumer.primary_input.signature, current_sigma)
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

        ctx = context or ExecutionContext()
        accumulated_sigma = Substitution()
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]] = []
        var_counter = getattr(ctx, "var_counter", 0)
        producer_var: Optional[str] = None

        # Clause decomposition of the prompt (connector-based fast path; shared with
        # the planner) — used solely for member-state affinity when projecting
        # heterogeneous products. Zero domain vocabulary.
        prompt_text = getattr(ctx, "prompt", "") or ""
        clause_token_sets: List[Set[str]] = []
        if prompt_text:
            for cl in re.split(r'[,;]|\b(?:and|then)\b', prompt_text.strip()):
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

        def _is_product(v_sig: Any) -> bool:
            raw = str(getattr(getattr(v_sig, "signature", None), "type_name", "") or "")
            return "[" in raw and raw.strip().lower().split("[", 1)[0] in ("tuple", "product", "pair")

        def _member_candidates(v_sig: Any, v_name: str, port_sig: Any, cell_toks: Set[str]) -> List[Tuple[float, int, Substitution]]:
            """Product-elimination candidates: (score, member_index, substitution)."""
            raw = str(getattr(getattr(v_sig, "signature", None), "type_name", "") or "")
            if "[" not in raw or not raw.endswith("]"):
                return []
            constructor = raw.split("[", 1)[0].strip().lower()
            if constructor not in ("tuple", "product", "pair"):
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
                except Exception:
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

        # Process each cell in sequence
        for idx, cell in enumerate(cells):
            # For-each replicas (multiplicity expansion): a replica RE-CONSUMES
            # its receiver from the environment's in-scope variables (e.g. the
            # source DataFrame) instead of the previous wire, and its reference
            # port binds the NEXT member of the identifier role group. It
            # consumes no incoming wire, exactly like a zero-ary constructor.
            is_replica = bool(getattr(cell, "replica_of", None))
            cell_bindings: Dict[str, str] = {}
            var_counter += 1
            ctx.var_counter = var_counter
            current_out_var = f"var_{var_counter}"
            cell_bindings["output_var"] = current_out_var
            ctx.consumed_tokens.update(t.lower() for t in cell.token_set)

            is_zero_ary = (
                getattr(cell, "node_type", "") == "constructor"
                and not any(p.required for p in cell.inputs.values())
            )

            # 1. If not the first cell, verify monadic transition from preceding cell.
            # Zero-ary constructors and replicas consume no incoming wire and skip the gate.
            if idx > 0 and not is_zero_ary and not is_replica:
                prev_cell = cells[idx - 1]
                transition_res = self.unify_transition(prev_cell, cell, accumulated_sigma, context=ctx)
                if transition_res.is_bottom():
                    return Failure(transition_res.reason if isinstance(transition_res, Failure) else "Transition failed")
                assert isinstance(transition_res, Success)
                accumulated_sigma = transition_res.sigma

            # 2. Multi-Port Monoidal Matching: Bind input ports across available wires
            req_ports = [p for p in cell.inputs.values() if p.required]
            bound_producer = False

            if len(req_ports) > 1 and len(ctx.variables) >= len(req_ports):
                # Symmetrical Monoidal Product Port Assignment (Section 3.2):
                # Search compatible wire assignments ensuring distinct wires for distinct required ports,
                # immediate producer wire connection (if producer_var is present), and maximal semantic affinity.
                avail_vars = list(ctx.variables.keys())
                best_assign = None
                best_sub = None
                best_affinity = -999.0

                import itertools
                for assignment in itertools.permutations(avail_vars, len(req_ports)):
                    if producer_var is not None and producer_var not in assignment:
                        continue
                    # A heterogeneous product (tuple[A,B,...]) is never assigned whole
                    # to a data port: its members must be projected individually.
                    if any(_is_product(ctx.variables[v_name][0]) for v_name in assignment):
                        continue
                    test_sub = accumulated_sigma
                    valid = True
                    aff_score = 0.0
                    for p, v_name in zip(req_ports, assignment):
                        v_sig, _ = ctx.variables[v_name]
                        u_p = unify(v_sig.signature, p.signature, test_sub)
                        if u_p is None:
                            valid = False
                            break
                        test_sub = u_p

                        # Token affinity between port name and variable's producing cell
                        p_toks = CellTokenizer.tokenize_identifier(p.name.lower())
                        src_cell = getattr(ctx, "var_sources", {}).get(v_name)
                        src_toks = set()
                        if src_cell:
                            src_toks = set(src_cell.token_set)
                            for slot_cells in getattr(src_cell, "bound_slots", {}).values():
                                for sc in slot_cells:
                                    src_toks.update(sc.token_set)
                        v_sig, _ = ctx.variables[v_name]
                        src_toks.update(CellTokenizer.tokenize_identifier(v_name.lower()))
                        if hasattr(v_sig, "name") and v_sig.name:
                            src_toks.update(CellTokenizer.tokenize_identifier(v_sig.name.lower()))
                        overlap = len(p_toks & src_toks)
                        aff_score += overlap * 2.0
                        if any(tok in src_toks for tok in p_toks):
                            aff_score += 1.0
                        if producer_var is not None and v_name == producer_var:
                            aff_score += 0.5

                    if valid and aff_score > best_affinity:
                        best_affinity = aff_score
                        best_assign = dict(zip([p.name for p in req_ports], assignment))
                        best_sub = test_sub

                if best_assign is not None:
                    for p_name, v_name in best_assign.items():
                        cell_bindings[p_name] = v_name
                    if best_sub is not None:
                        accumulated_sigma = best_sub
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

            # If producer_var did not bind to primary input, check other compatible input ports
            if producer_var is not None and not bound_producer and not is_zero_ary and not is_replica:
                prod_sig, _ = ctx.variables.get(producer_var, (None, None))
                if prod_sig is not None and not _is_product(prod_sig):
                    for p_name, p_sig in cell.inputs.items():
                        if p_name not in cell_bindings:
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

                # Optional parameters with declared default: use prompt literal or default value
                if not p_sig.required and p_sig.default_value is not None:
                    resolved_literal = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()))
                    val = resolved_literal if resolved_literal is not None else str(p_sig.default_value)
                    cell_bindings[p_name] = val
                    accumulated_sigma.bind(p_name, val)
                    continue

                # Optional parameters WITHOUT a default are omitted from the
                # emitted call unless the prompt itself supplies a literal through
                # the type-affinity channels (a destination path, a numeric or
                # boolean qualifier). They capture neither in-scope data wires nor
                # the expression channel: configuration slots are not dataflow
                # continuations, and the shared expression channel belongs to the
                # cell's required data ports.
                if not p_sig.required:
                    _lit = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()))
                    if _lit is not None:
                        cell_bindings[p_name] = _lit
                        accumulated_sigma.bind(p_name, _lit)
                    else:
                        cell_bindings[p_name] = None
                    continue

                # A. Check in-scope variables first (environment / predecessor variables matching typestate)
                scoped_var = None
                for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                    # Heterogeneous products project member-wise; the whole container
                    # is never wired into a single data port.
                    if _is_product(v_sig):
                        continue
                    u_v = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                    if u_v is not None:
                        scoped_var = v_name
                        accumulated_sigma = u_v
                        break

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

                # B. Check typestate-driven literal resolution from prompt
                resolved_literal = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()))
                if resolved_literal is not None:
                    cell_bindings[p_name] = resolved_literal
                    accumulated_sigma.bind(p_name, resolved_literal)
                    continue

                # C. Check default value declared in tree
                if p_sig.default_value is not None:
                    cell_bindings[p_name] = str(p_sig.default_value)
                    accumulated_sigma.bind(p_name, str(p_sig.default_value))
                    continue

                # D. Declared-domain enum grounding (reflection-driven, zero domain hardcodes)
                p_domain = getattr(concrete_sig, "domain", "") or ""
                if p_domain and getattr(ctx, "prompt", ""):
                    enum_val = ctx._resolve_enum_constant(p_domain, concrete_sig.state)
                    if enum_val is not None:
                        cell_bindings[p_name] = enum_val
                        accumulated_sigma.bind(p_name, enum_val)
                        continue

                # E. Unresolved REQUIRED port: fail loudly. A pipeline with an
                # unsatisfiable required port must not emit garbage code (e.g. a bare
                # identifier that NameErrors at runtime); it is reported as a synthesis
                # failure so the caller (or the LLM repair cycle) can react honestly.
                if p_sig.required:
                    raise UnresolvedPlaceholderError(
                        f"Required port '{p_name}' of cell '{cell.cell_id}' "
                        f"(type '{concrete_sig.type_name}', state '{concrete_sig.state}') "
                        f"could not be resolved from the prompt, context, or declared defaults."
                    )
                else:
                    cell_bindings[p_name] = None

            # Register output port in context for future steps
            # Concrete output port with generic substitution
            concrete_out = substitute_generics(cell.primary_output, accumulated_sigma)
            ctx.declare_variable(current_out_var, concrete_out, current_out_var, cell=cell)
            producer_var = current_out_var
            pipeline_bindings.append((cell, cell_bindings))

        return Success(pipeline_bindings, accumulated_sigma)

    @staticmethod
    def _instantiate_ast_template(template: str, bindings: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """
        Synthesizes executable Python code from an AST template, adhering to identity omission semantics:
        - Required positional parameters are instantiated with their bound values.
        - Actively bound optional configurations are emitted as keyword arguments (key=val).
        - Unbound optional parameters with defaults are omitted, letting runtime defaults apply.
        """
        if not template or not template.strip():
            return ""

        ph_map: Dict[str, str] = {}
        def to_ph(m):
            name = m.group(1)
            ph_id = f"_nstl_ph_{name}"
            ph_map[ph_id] = name
            return ph_id

        ast_ready = re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", to_ph, template)
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
                new_keywords = list(node.keywords)
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in ph_map:
                        orig_name = ph_map[arg.id]
                        p_sig = inputs.get(orig_name)
                        is_req = getattr(p_sig, "required", True) if p_sig else True
                        val = bindings.get(orig_name)

                        if val is not None:
                            val_str = str(val)
                            try:
                                val_node = ast.parse(val_str, mode="eval").body
                            except Exception:
                                val_node = ast.Constant(value=val_str)
                            if is_req:
                                new_args.append(val_node)
                            else:
                                new_keywords.append(ast.keyword(arg=orig_name, value=val_node))
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
        except Exception:
            res = template
            for k, v in bindings.items():
                if v is not None:
                    res = res.replace(f"{{{k}}}", str(v))
            return res

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
        accum_sigma: Substitution = Substitution()
        if cells and isinstance(cells[0], tuple):
            pipeline_bindings = cells
        else:
            res = self.unify_pipeline(cells, ctx)
            if res.is_bottom():
                reason = res.reason if isinstance(res, Failure) else "Unknown unification failure"
                raise ValueError(f"Unification Failed: {reason}")
            assert isinstance(res, Success)
            pipeline_bindings = res.value
            accum_sigma = res.sigma
            self.last_egress_paths = self._derive_egress_paths(pipeline_bindings)

        # Collect dependencies recursively
        deps: List[str] = []
        def collect_deps(c: Cell):
            for dep in c.dependencies:
                dep_clean = dep.strip()
                if dep_clean and dep_clean not in deps:
                    deps.append(dep_clean)
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
            if has_render_cell and (getattr(cell, "bound_slots", None) or getattr(cell, "slots", None)):
                rendered = render_cell(cell, bindings, indent_level=0, context=ctx, accumulated_sigma=accum_sigma)
                if rendered:
                    code_lines.append(rendered)
            else:
                template = cell.code_template.strip()
                if not template:
                    continue

                instantiated = self._instantiate_ast_template(template, bindings, cell.inputs)
                code_lines.append(instantiated)

        final_code = "\n".join(code_lines).strip()
        return final_code

    def unify_and_emit(self, cells: List[Cell], prompt: str = "") -> str:
        """Main synthesis entrypoint."""
        self.context = ExecutionContext(prompt=prompt)
        return self.emit_code(cells, self.context)

    @classmethod
    def unify_cell(cls, context: Any, cell: Cell) -> str:
        """Single-cell unification helper for backwards-compatibility with tests."""
        gate = cls()
        ctx = context if isinstance(context, ExecutionContext) else ExecutionContext(str(context))
        return gate.emit_code([cell], ctx)

    @classmethod
    def resolve_imports(cls, code_text: str, context: Any = None, chain_nodes: Any = None) -> Union[str, List[str]]:
        """Collects declared dependencies strictly from chain_nodes without domain hardcodes."""
        imports = set()
        if chain_nodes:
            for node in chain_nodes:
                for dep in getattr(node, "dependencies", []):
                    dep_str = dep.strip()
                    if dep_str:
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


TOP_TYPE_SET = {"any", "Any", "*", "top", "⊤", "object", "Object"}


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
        # Check in-scope variables
        if hasattr(ctx, "scope_variables") and ctx.scope_variables:
            return list(ctx.scope_variables.keys())[-1]

        # Check default value
        if getattr(port_sig, "default_value", None) is not None:
            return str(port_sig.default_value)

        # Check context parameters
        if hasattr(ctx, "parameters") and port_name in ctx.parameters:
            return str(ctx.parameters[port_name])

        return current_out_var


PlaceholderResolver = DynamicPlaceholderResolver


@dataclass
class ExtractedSlots:
    source_uris: List[str] = field(default_factory=list)
    dest_uris: List[str] = field(default_factory=list)
    named_identifiers: List[str] = field(default_factory=list)
    numeric_literals: List[Union[int, float]] = field(default_factory=list)
    operational_flags: Dict[str, Any] = field(default_factory=dict)

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
        }


class ParameterExtractor:
    """Compatibility adapter over ExecutionContext."""
    @staticmethod
    def extract_slots(prompt: str) -> ExtractedSlots:
        ctx = ExecutionContext(prompt=prompt)
        slots = ExtractedSlots()
        file_assets = [val for _, kind, val in ctx.ordered_literals if kind in ("file_asset", "quoted_str") and "." in val]
        if file_assets:
            slots.source_uris = [file_assets[0]]
            slots.dest_uris = file_assets[1:]
        slots.numeric_literals = [
            float(val) if "." in val else int(val)
            for _, kind, val in ctx.ordered_literals if kind == "numeric"
        ]
        quoted_strings = [val for _, kind, val in ctx.ordered_literals if kind == "quoted_str" and "." not in val]
        if quoted_strings:
            slots.named_identifiers.extend(quoted_strings)
        else:
            sym = ctx._project_semantic_slot()
            if sym:
                slots.named_identifiers.append(sym)
        return slots

    @staticmethod
    def extract_parameters(prompt: str) -> Dict[str, Any]:
        return ParameterExtractor.extract_slots(prompt).to_dict()


def enforce_lineage_integrity(code: str, target_cells=None) -> str:
    """Compatibility passthrough for lineage tracking."""
    return code
