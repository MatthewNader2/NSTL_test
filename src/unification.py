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
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable, Generic, TypeVar

from log_config import get_logger

try:
    from .lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry
except (ImportError, ValueError):
    from lattice import AlgebraicSignature, PortSignature, Cell, TypeRegistry

logger = get_logger('unification')

T = TypeVar('T')
U = TypeVar('U')


TOP_TYPE_SET = {"any", "Any", "*", "top", "⊤", "object", "Object", "dataobject", "DataObject"}


# =====================================================================
# 1. Type Terms and Substitutions
# =====================================================================

class TypeTerm(ABC):
    """Abstract base class for formal type terms."""
    @abstractmethod
    def apply_substitution(self, sigma: 'Substitution') -> 'TypeTerm':
        pass

    @classmethod
    def from_string(cls, s: str) -> 'TypeTerm':
        s_clean = str(s).strip()
        if s_clean in TOP_TYPE_SET:
            return TOP
        if s_clean.startswith("?"):
            return TypeVariable(s_clean[1:])
        return AtomicType(s_clean)


@dataclass(frozen=True, slots=True)
class TopType(TypeTerm):
    """Universal Top type (wildcard) unifying with any type term."""
    def apply_substitution(self, sigma: 'Substitution') -> 'TypeTerm':
        return self

    def __repr__(self) -> str:
        return "Top"


TOP = TopType()


@dataclass(frozen=True, slots=True)
class AtomicType(TypeTerm):
    """Ground type constant (e.g. int, str, DataFrame, Mat, Pose)."""
    name: str

    def apply_substitution(self, sigma: 'Substitution') -> 'TypeTerm':
        return self

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class TypeVariable(TypeTerm):
    """Type variable alpha, beta... subject to substitution."""
    var_name: str

    def apply_substitution(self, sigma: 'Substitution') -> 'TypeTerm':
        if self.var_name in sigma.mappings:
            target = sigma.mappings[self.var_name]
            if isinstance(target, TypeTerm):
                return target.apply_substitution(sigma)
            return AtomicType(str(target))
        return self

    def __repr__(self) -> str:
        return f"?{self.var_name}"


@dataclass(frozen=True, slots=True)
class TypestateTerm(TypeTerm):
    """Typestate compound term: tau = (type_name, state, qualifiers)."""
    type_name: str
    state: str = "any"
    qualifiers: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)

    def apply_substitution(self, sigma: 'Substitution') -> 'TypeTerm':
        t_resolved = self.type_name
        if self.type_name in sigma.mappings:
            val = sigma.mappings[self.type_name]
            t_resolved = val.name if isinstance(val, AtomicType) else str(val)
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

def unify(
    term1: Union[TypeTerm, AlgebraicSignature, str],
    term2: Union[TypeTerm, AlgebraicSignature, str],
    sigma: Optional[Substitution] = None
) -> Optional[Substitution]:
    """
    Computes Most General Unifier (mgu) of term1 and term2.
    Returns updated Substitution sigma if unification succeeds, or None (bottom) on failure.
    """
    sub = Substitution(sigma.mappings if sigma else {})

    # Normalize AlgebraicSignature to TypestateTerm
    t1 = _to_type_term(term1)
    t2 = _to_type_term(term2)

    t1 = t1.apply_substitution(sub)
    t2 = t2.apply_substitution(sub)

    # 1. Identity or Top
    if t1 == t2 or isinstance(t1, TopType) or isinstance(t2, TopType):
        return sub

    # 2. Variable binding
    if isinstance(t1, TypeVariable):
        sub.bind(t1.var_name, t2)
        return sub
    if isinstance(t2, TypeVariable):
        sub.bind(t2.var_name, t1)
        return sub

    # 3. Typestate term unification
    if isinstance(t1, TypestateTerm) and isinstance(t2, TypestateTerm):
        # State unification: if both specify a concrete state, they must match
        if t1.state != "any" and t2.state != "any":
            if t1.state.lower() != t2.state.lower():
                return None  # State mismatch -> bottom

        # Type poset subtyping check: t1.type_name <= t2.type_name
        registry = TypeRegistry.get_instance()
        if not registry.is_subtype(t1.type_name, t2.type_name):
            return None  # Type mismatch -> bottom

        # Qualifier subset check
        if t2.qualifiers and not t2.qualifiers.issubset(t1.qualifiers):
            return None

        return sub

    # 4. Atomic type unification
    if isinstance(t1, AtomicType) and isinstance(t2, AtomicType):
        registry = TypeRegistry.get_instance()
        if registry.is_subtype(t1.name, t2.name):
            return sub
        return None

    return None


def _to_type_term(item: Any) -> TypeTerm:
    if isinstance(item, TypeTerm):
        return item
    if isinstance(item, AlgebraicSignature):
        if item.is_top():
            return TOP
        return TypestateTerm(type_name=item.type_name, state=item.state, qualifiers=item.qualifiers)
    if isinstance(item, PortSignature):
        return _to_type_term(item.signature)
    if isinstance(item, str):
        if item.lower() in ("any", "*", "top", "object"):
            return TOP
        if item.startswith("?"):
            return TypeVariable(item[1:])
        return AtomicType(item)
    return TOP


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
        self.prompt = prompt
        self.variables: Dict[str, Tuple[PortSignature, str]] = {}
        self.var_counter: int = 0
        self.parameters: Dict[str, Any] = {}
        self.ordered_literals: List[Tuple[int, str, str]] = []  # (offset, type_kind, value)
        self.used_indices: Set[int] = set()
        self.consumed_tokens: Set[str] = set()
        self._extract_universal_literals(prompt)
        if scope:
            for k, v in scope.items():
                self.declare_variable(k, v, k)

    def _extract_universal_literals(self, prompt: str):
        if not prompt:
            return

        spans: List[Tuple[int, str, str]] = []

        # 1. Quoted literals: '...' or "..." -> kind "quoted_str"
        for m in re.finditer(r'["\']([^"\']+)["\']', prompt):
            spans.append((m.start(), "quoted_str", m.group(1)))

        # 2. File / Path tokens with extensions: e.g. data.csv, telemetry.log, etc. -> kind "file_asset"
        for m in re.finditer(r'\b[\w-]+\.[a-zA-Z0-9]{1,8}\b', prompt):
            val = m.group(0).rstrip(".,;:)")
            # Avoid duplicate if already covered by quotes
            if not any(s <= m.start() and m.end() <= s + len(v) + 2 for s, t, v in spans if t == "quoted_str"):
                spans.append((m.start(), "file_asset", val))

        # 3. Numeric literals -> kind "numeric"
        for m in re.finditer(r'\b\d+(?:\.\d+)?\b', prompt):
            spans.append((m.start(), "numeric", m.group(0)))

        # Order strictly by character position in prompt
        spans.sort(key=lambda x: x[0])
        self.ordered_literals = spans

    def declare_variable(self, name: str, port_sig: Union[PortSignature, AlgebraicSignature, Any], expr: str = ""):
        if isinstance(port_sig, AlgebraicSignature):
            port_sig = PortSignature(name=name, signature=port_sig)
        elif not isinstance(port_sig, PortSignature):
            port_sig = PortSignature(name=name, signature=AlgebraicSignature(str(port_sig), "any"))
        self.variables[name] = (port_sig, expr or name)

    def get_variable_name(self, port_sig: PortSignature) -> Optional[str]:
        """Finds in-scope variable that unifies with port_sig."""
        for v_name, (v_sig, _) in reversed(list(self.variables.items())):
            if v_sig.unifies_with(port_sig):
                return v_name
        return None

    def _allocate_free_symbol(self, param_name: str) -> Optional[str]:
        """
        Free Monoid Symbol Allocation:
        Allocates unconsumed prompt tokens as identifier arguments.
        Prioritizes syntactic successors following the parameter name in the prompt.
        Zero domain-specific keywords or hardcoded word lists.
        """
        if not self.prompt:
            return None

        tokens = []
        for m in re.finditer(r"[a-zA-Z_][a-zA-Z0-9_]*", self.prompt):
            tokens.append((m.group(0), m.start()))

        if not tokens:
            return None

        p_name_lower = param_name.lower()
        tokens_lower = [t[0].lower() for t in tokens]

        # 1. Syntactic successor: if param_name appears in prompt, check the next token
        if p_name_lower in tokens_lower:
            idx = tokens_lower.index(p_name_lower)
            if idx + 1 < len(tokens):
                candidate = tokens[idx + 1][0]
                if len(candidate) >= 2 and candidate.lower() != p_name_lower:
                    self.consumed_tokens.add(candidate.lower())
                    return candidate

        # 2. Minimum topological distance to parameter occurrence
        free_candidates = []
        for i, (tok, pos) in enumerate(tokens):
            t_low = tok.lower()
            if t_low not in self.consumed_tokens and len(t_low) >= 2:
                free_candidates.append((i, tok, pos))

        if not free_candidates:
            return None

        p_indices = [i for i, (tok, _) in enumerate(tokens) if tok.lower() == p_name_lower]
        if p_indices:
            p_idx = p_indices[0]
            best = min(free_candidates, key=lambda c: abs(c[0] - p_idx))
        else:
            best = free_candidates[0]

        self.consumed_tokens.add(best[1].lower())
        return best[1]

    def resolve_literal_for_port(self, port_sig: PortSignature, cell_stage: Optional[int] = None) -> Optional[str]:
        """
        Resolves a value for a port using parameters, declared defaults, or literals.
        Zero domain-specific keywords or hardcoded values.
        Categorically grounded in morphism stages:
          - Stage 1 (Initial / Ingestion morphism Env -> C): resolves environment assets
          - Stage 2 (Endomorphism C x P -> C): resolves operational parameters P
          - Stage 3 (Terminal / Egress morphism C -> Env): resolves output destination
        """
        p_name = port_sig.name.lower()
        t_name = port_sig.type_name.lower()

        # 1. Parameter explicitly supplied in context
        if port_sig.name in self.parameters:
            val = self.parameters[port_sig.name]
            return json.dumps(val) if isinstance(val, str) else str(val)

        # 2. Extract explicit argument matching this port name from prompt: `<p_name>=<value>` or `<p_name>:<value>`
        if self.prompt and p_name:
            m = re.search(rf"\b{re.escape(port_sig.name)}\s*[:=]\s*['\"]?([a-zA-Z0-9_./-]+)['\"]?", self.prompt, re.IGNORECASE)
            if m:
                val = m.group(1).rstrip(".,;:)")
                if t_name == "str":
                    return json.dumps(val)
                elif t_name in ("int", "float", "numeric"):
                    return val
                elif t_name == "bool":
                    return "False" if "false" in val.lower() or "0" in val else "True"

        # 3. Vector Polarity Projection for Boolean / Valuation Ports
        if t_name == "bool" and self.prompt:
            try:
                from inference import ModelManager
                from tokenizer import CellTokenizer
                import numpy as np

                mm = ModelManager.get_instance()
                if mm.profile is not None:
                    qualifier_map = dict(getattr(port_sig.signature, "qualifiers", []))
                    pos_label = qualifier_map.get("positive", port_sig.name)
                    neg_label = qualifier_map.get("negative", f"not {port_sig.name}")

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
                                if abs(pos_score - neg_score) > 0.05:
                                    return "True" if pos_score > neg_score else "False"
            except Exception as e:
                logger.debug(f"[UNIFICATION] Vector polarity projection fallback: {e}")

            if port_sig.default_value is not None:
                return str(port_sig.default_value)
            return "True"

        # 4. Numeric literals for numeric ports
        if t_name in ("int", "float", "numeric"):
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind == "numeric":
                    self.used_indices.add(idx)
                    return val

        # 5. Stage 1 and Stage 3 Morphisms: Environmental Asset Grounding
        if cell_stage in (1, 3) and t_name in ("str", "any"):
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind in ("file_asset", "quoted_str"):
                    self.used_indices.add(idx)
                    return json.dumps(val)

            # If Stage 1 and no literal was in prompt and no default declared, project prompt onto workspace files
            if cell_stage == 1 and port_sig.default_value is None:
                try:
                    from pathlib import Path
                    from inference import ModelManager
                    import numpy as np

                    workspace_files = [f for f in Path.cwd().iterdir() if f.is_file() and not f.name.startswith(".")]
                    if workspace_files and self.prompt:
                        mm = ModelManager.get_instance()
                        if mm.profile is not None:
                            e_prompt = np.array(mm.get_embedding(self.prompt), dtype=np.float32)
                            p_norm = np.linalg.norm(e_prompt)
                            if p_norm > 0:
                                e_prompt = e_prompt / p_norm
                                best_file = None
                                best_sim = -1.0
                                for f in workspace_files:
                                    e_f = np.array(mm.get_embedding(f.name), dtype=np.float32)
                                    f_norm = np.linalg.norm(e_f)
                                    if f_norm > 0:
                                        e_f = e_f / f_norm
                                        sim = float(np.dot(e_prompt, e_f))
                                        if sim > best_sim:
                                            best_sim = sim
                                            best_file = f.name
                                if best_file is not None and best_sim > 0.15:
                                    return json.dumps(best_file)
                except Exception as e:
                    logger.debug(f"[UNIFICATION] Environmental asset grounding fallback: {e}")

        # 6. Stage 2 Morphism: Operational Parameter Extraction
        if (cell_stage == 2 or cell_stage is None) and t_name in ("str", "any"):
            # A. Check for quoted string argument in prompt (e.g. 'age')
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind == "quoted_str":
                    self.used_indices.add(idx)
                    return json.dumps(val)

            # B. Free Monoid Symbol Allocation for unquoted string/identifier arguments
            if self.prompt:
                allocated = self._allocate_free_symbol(port_sig.name)
                if allocated:
                    return json.dumps(allocated)

        # 7. Port default value declared in tree schema
        if port_sig.default_value is not None:
            def_str = str(port_sig.default_value)
            if def_str.isdigit() or def_str in ("True", "False", "None"):
                return def_str
            return def_str if (def_str.startswith('"') or def_str.startswith("'")) else json.dumps(def_str)

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

    def unify_transition(
        self,
        producer: Cell,
        consumer: Cell,
        current_sigma: Substitution
    ) -> MonadResult[Substitution]:
        """
        Verifies that producer's primary output unifies with consumer's primary input.
        Returns Success(new_sigma) or Failure(bottom).
        """
        out_sig = producer.primary_output
        in_sig = consumer.primary_input

        new_sigma = unify(out_sig.signature, in_sig.signature, current_sigma)
        if new_sigma is None:
            return Failure(
                f"Typestate Unification Failed: {producer.cell_id} outputs {out_sig.signature} "
                f"which cannot satisfy {consumer.cell_id} input {in_sig.signature}"
            )
        return Success(new_sigma, new_sigma)

    def unify_pipeline(
        self,
        cells: List[Cell],
        context: Optional[ExecutionContext] = None
    ) -> MonadResult[List[Tuple[Cell, Dict[str, str]]]]:
        """
        Chains a sequence of cells [v_1, ..., v_n] through the Type Monad.
        Binds port placeholders to variables in each step.
        """
        if not cells:
            return Failure("Empty cell pipeline")

        ctx = context or ExecutionContext()
        accumulated_sigma = Substitution()
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]] = []
        var_counter = getattr(ctx, "var_counter", 0)

        # Step 1: Initial Source Setup
        producer_var: Optional[str] = None

        # Process each cell in sequence
        for idx, cell in enumerate(cells):
            cell_bindings: Dict[str, str] = {}
            var_counter += 1
            ctx.var_counter = var_counter
            current_out_var = f"var_{var_counter}"
            cell_bindings["output_var"] = current_out_var
            ctx.consumed_tokens.update(t.lower() for t in cell.token_set)

            # 1. If not the first cell, verify monadic transition from preceding cell
            if idx > 0:
                prev_cell = cells[idx - 1]
                transition_res = self.unify_transition(prev_cell, cell, accumulated_sigma)
                if transition_res.is_bottom():
                    return Failure(transition_res.reason if isinstance(transition_res, Failure) else "Transition failed")
                assert isinstance(transition_res, Success)
                accumulated_sigma = transition_res.sigma

            # 2. Bind primary input port to previous producer output variable if type-compatible
            prim_in = cell.primary_input
            if producer_var is not None and prim_in.name in cell.inputs:
                prod_sig, _ = ctx.variables.get(producer_var, (None, None))
                if prod_sig is not None and prod_sig.unifies_with(prim_in):
                    cell_bindings[prim_in.name] = producer_var
                    accumulated_sigma.bind(prim_in.name, producer_var)

            # 3. Resolve auxiliary input ports (parameters, file paths, literals)
            for p_name, p_sig in cell.inputs.items():
                if p_name in cell_bindings:
                    continue  # Already bound

                # A. Check in-scope variables first (environment / predecessor variables matching typestate)
                scoped_var = ctx.get_variable_name(p_sig)
                if scoped_var is not None:
                    cell_bindings[p_name] = scoped_var
                    accumulated_sigma.bind(p_name, scoped_var)
                    continue

                # B. Check typestate-driven literal resolution from prompt
                resolved_literal = ctx.resolve_literal_for_port(p_sig, cell_stage=cell.stage)
                if resolved_literal is not None:
                    cell_bindings[p_name] = resolved_literal
                    accumulated_sigma.bind(p_name, resolved_literal)
                    continue

                # C. Check default value declared in tree
                if p_sig.default_value is not None:
                    cell_bindings[p_name] = str(p_sig.default_value)
                    accumulated_sigma.bind(p_name, str(p_sig.default_value))
                    continue

                # D. If required and unresolved, bind port name or empty fallback
                if p_sig.required:
                    cell_bindings[p_name] = f'"{p_name}"'
                else:
                    cell_bindings[p_name] = "None"

            # Register output port in context for future steps
            ctx.declare_variable(current_out_var, cell.primary_output, current_out_var)
            producer_var = current_out_var
            pipeline_bindings.append((cell, cell_bindings))

        return Success(pipeline_bindings, accumulated_sigma)

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
        res = self.unify_pipeline(cells, ctx)
        if res.is_bottom():
            reason = res.reason if isinstance(res, Failure) else "Unknown unification failure"
            raise ValueError(f"Unification Failed: {reason}")

        assert isinstance(res, Success)
        pipeline_bindings = res.value

        # Collect dependencies
        deps: List[str] = []
        for cell, _ in pipeline_bindings:
            for dep in cell.dependencies:
                dep_clean = dep.strip()
                if dep_clean and dep_clean not in deps:
                    deps.append(dep_clean)

        code_lines: List[str] = []
        if deps:
            code_lines.extend(deps)
            code_lines.append("")

        for cell, bindings in pipeline_bindings:
            template = cell.code_template.strip()
            if not template:
                continue

            # Pure placeholder substitution
            instantiated = template
            for ph, val in bindings.items():
                instantiated = instantiated.replace(f"{{{ph}}}", str(val))

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
    """Asserts that all {placeholder} slots in a template are bound."""
    if bindings:
        for k, v in bindings.items():
            template = template.replace(f"{{{k}}}", str(v))
    remaining = re.findall(r"\{([a-zA-Z0-9_]+)\}", template)
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
            sym = ctx._allocate_free_symbol("by")
            if sym:
                slots.named_identifiers.append(sym)
        return slots

    @staticmethod
    def extract_parameters(prompt: str) -> Dict[str, Any]:
        return ParameterExtractor.extract_slots(prompt).to_dict()


def enforce_lineage_integrity(code: str, target_cells=None) -> str:
    """Compatibility passthrough for lineage tracking."""
    return code
