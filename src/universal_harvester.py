"""
src/universal_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Unified Harvest Pipeline over adaptive library adapters.

Architecture:
  ONE pipeline owns all schema semantics (stage classification, typestate
  modeling, morphism generation). LibraryAdapter instances (see
  library_adapters.py) are selected by the IMPLEMENTATION KIND of the target
  library — pure-Python source, typing-stubbed extension, or runtime-reflection
  only — and supply the raw evidence channels the pipeline asks for. Libraries
  built the same way are served by the same adapter; no library names,
  no domain heuristics, no hardcoded types.

Stateful-lifecycle modeling (typestate correctness):
  The pipeline models mutating instance methods as monadic ENDOMORPHISMS on
  the receiver's own type:  C[any] -> C[mutated].  A mutator is detected
  structurally, by evidence, in richness order:
    1. return annotation  -> Self / "Self" / the class itself
    2. bare `return self` in the method source (fluent-lifecycle convention)
    3. explicit -> None annotation on a bound instance method
       (the Python in-place convention)
  For every class that owns at least one detected mutator, the pipeline runs
  an AST attribute-dataflow analysis: a method whose source READS an instance
  attribute that a mutator WRITES — or that transitively calls such a method
  through `self` — is STATE-DEPENDENT, and its receiver port declares
  C[mutated]. The existing typestate unification then enforces the full
  lifecycle (construct -> mutate -> use) automatically, corpus-wide, for
  every stateful class in every pure-Python library. Stub-only libraries keep
  receiver state "any" (no evidence -> no fabricated constraints).
"""

from __future__ import annotations

import ast as _ast
import enum
import importlib
import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings

# Suppress noisy reflection warnings from third-party library namespaces
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from schema import CellSchema, PortSchema, TreeSchema
    from log_config import get_logger
    from tokenizer import CellTokenizer
    from signature_introspector import (
        resolve_signature,
        get_callable_parameters,
        extract_clean_type_name
    )
    from library_adapters import get_adapter_for_package
except ImportError:
    from .schema import CellSchema, PortSchema, TreeSchema
    from .log_config import get_logger
    from .tokenizer import CellTokenizer
    from .signature_introspector import (
        resolve_signature,
        get_callable_parameters,
        extract_clean_type_name
    )
    from .library_adapters import get_adapter_for_package

logger = get_logger("universal_harvester")

# Universal language-level base primitives (universal across Python, not specific to any library)
UNIVERSAL_BASE_PRIMITIVES: Set[str] = {
    "int", "float", "str", "bool", "bytes", "bytearray",
    "none", "nonetype", "void", "noreturn", "any", "object",
    "tuple", "list", "dict", "set", "frozenset",
    "iterable", "iterator", "generator", "sequence", "mapping",
    "callable", "type", "ellipsis"
}

# Declared typestate vocabulary (language-universal lifecycle states)
CONSTRUCTED_STATE = "constructed"
MUTATED_STATE = "mutated"

_NONE_RETURNS = ("none", "nonetype", "void", "noreturn")
_WILDCARD_RETURNS = ("any", "", "*", "top", "unknown")


def _is_constant_like(attr_name: str, val: Any) -> bool:
    """Universal structural check for constants (PEP 8 convention or Enum instance)."""
    return attr_name.isupper() or isinstance(val, enum.Enum)


def _is_boilerplate_config_method(params: List[inspect.Parameter]) -> bool:
    """
    Structural boilerplate filter: a method whose ONLY parameters are keyword-only
    with defaults (or **kwargs) and which declares no positional data ports is a
    configuration/metadata mutator (e.g. the framework-generated
    ``set_<step>_request``/``set_output`` families), never a dataflow intent.
    Zero name patterns: the decision is purely signature-structural.
    """
    if not params:
        return False
    for p in params:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty:
            continue
        return False
    return True


def _returns_self_in_source(source: Optional[str]) -> bool:
    """
    Structural self-return detection: scans the function SOURCE for a bare
    `return self` statement (fluent/mutator lifecycle methods, e.g. estimator
    .fit). Pure text-level scan over adapter-provided source: zero regex,
    zero name patterns. Returns False when source is unavailable.
    """
    if not source:
        return False
    for line in source.splitlines():
        s = line.strip()
        if s.startswith("return "):
            expr = s[len("return "):].strip().rstrip(";").strip()
            if expr == "self":
                return True
    return False


def _public_alias(module_path: str, attr_name: str) -> str:
    """
    Resolves the most public importable path for a callable/class.
    If the defining module path contains a private segment (``._x``), walks up
    to the first parent package that re-exports the attribute and returns that
    shallow path; returns the original path when no shallower alias exists.
    Keeps emitted code templates on the library's public import surface.
    """
    path = module_path
    while "._" in path:
        path = path.split("._", 1)[0]
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, attr_name):
                return f"{path}.{attr_name}"
        except Exception:
            continue
    return f"{module_path}.{attr_name}"


def _module_priority(module_name: str) -> int:
    """
    Module-visibility tiering (source_priority): the shallower a defining module
    sits in the package namespace, the more public and documented its surface.
    Depth 1 (top-level) = 10, depth 2 = 40, depth >= 3 = 90. Deep/internal
    utilities therefore lose lexical ties against canonical public API cells,
    while remaining available as last-resort fallbacks.
    """
    depth = module_name.count(".")
    if depth <= 0:
        return 10
    if depth == 1:
        return 40
    return 90


def _is_fully_untyped(inputs: Dict[str, Any], out_type: str) -> bool:
    """
    True iff every input port and the output port carry the wildcard carrier.
    Fully-untyped morphisms carry no verifiable dataflow semantics and are
    demoted so typed alternatives always win routing ties.
    """
    if str(out_type).lower() not in ("any", "", "none", "*"):
        return False
    for p in inputs.values():
        t = getattr(p, "type_name", None) if not isinstance(p, dict) else p.get("type_name")
        if str(t).lower() not in ("any", "", "none", "*"):
            return False
    return True


def _harvest_constant_cell(
    domain_name: str,
    cid: str,
    m_name: str,
    code_expr: str,
    val: Any,
    tokens: List[str],
    dependencies: List[str],
) -> CellSchema:
    val_type = type(val).__name__
    return CellSchema(
        cell_id=cid,
        stage=1,
        inputs={},
        outputs={"value": PortSchema(type_name=val_type, state="constant")},
        code_template=code_expr,
        dependencies=dependencies,
        semantic_tags=tokens,
        keywords=tokens,
        docstring=f"Constant {code_expr}",
        domain_name=domain_name,
        node_type="constant",
        node_role="constant",
        source_priority=100
    )


def _safe_getattr(obj: Any, attr_name: str) -> Optional[Any]:
    """
    Safely retrieves an attribute. If accessing the attribute triggers a
    DeprecationWarning or FutureWarning, returns None so NSTL skips the
    deprecated alias and only harvests the canonical version.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            val = getattr(obj, attr_name, None)
        except Exception:
            return None

        # If accessing it triggers a deprecation warning, skip this alias
        if any(issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)) for w in caught):
            return None

        return val


# =====================================================================
# AST instance-attribute dataflow evidence (stateful lifecycle analysis)
# =====================================================================

def _parse_method_source(source: Optional[str]) -> Optional[_ast.Module]:
    """Parses adapter-provided source; class-body snippets are dedented first."""
    if not source:
        return None
    try:
        return _ast.parse(textwrap.dedent(source))
    except Exception:
        return None

def _self_attribute_writes(source: str, method_names: Set[str]) -> Set[str]:
    """
    Instance attributes WRITTEN in a method body: `self.X = ...`, `del self.X`,
    and the setattr idiom used by classes that override __setattr__
    (pandas: object.__setattr__(self, "_mgr", ...)). Method names are excluded
    (writing a method attribute is monkey-patching, not state).
    """
    tree = _parse_method_source(source)
    if tree is None:
        return set()
    writes: Set[str] = set()
    for node in _ast.walk(tree):
        if (
            isinstance(node, _ast.Attribute)
            and isinstance(node.value, _ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, (_ast.Store, _ast.Del))
        ):
            if node.attr not in method_names:
                writes.add(node.attr)
        elif (
            isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Attribute)
            and node.func.attr in ("__setattr__", "setattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], _ast.Constant)
            and isinstance(node.args[1].value, str)
            and any(isinstance(a, _ast.Name) and a.id == "self" for a in _ast.walk(node.args[0]))
        ):
            if node.args[1].value not in method_names:
                writes.add(node.args[1].value)
    return writes


def _self_attribute_reads(source: str, method_names: Set[str]) -> Set[str]:
    """Instance attributes READ via `self.X` (Load context), excluding method
    accesses (those are call dependencies, tracked separately)."""
    tree = _parse_method_source(source)
    if tree is None:
        return set()
    reads: Set[str] = set()
    for node in _ast.walk(tree):
        if (
            isinstance(node, _ast.Attribute)
            and isinstance(node.value, _ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, _ast.Load)
        ):
            if node.attr not in method_names:
                reads.add(node.attr)
    return reads


def _self_method_calls(source: str) -> Set[str]:
    """Names of methods invoked through `self.X(...)` — the intra-class call graph."""
    tree = _parse_method_source(source)
    if tree is None:
        return set()
    calls: Set[str] = set()
    for node in _ast.walk(tree):
        if (
            isinstance(node, _ast.Call)
            and isinstance(node.func, _ast.Attribute)
            and isinstance(node.func.value, _ast.Name)
            and node.func.value.id == "self"
        ):
            calls.add(node.func.attr)
    return calls


class UniversalHarvester:
    """
    Unified Harvest Pipeline.

    Stage A (adapter selection): evidence adapters are chosen by the library's
    implementation kind (source / stubs / runtime) — never by name.

    Stage B (enumeration): constants, classes (constructors, class constants,
    instance methods, static/class methods), and module-level functions.

    Stage C (typestate modeling): structural morphism classification plus
    mutator/endomorphism detection and receiver-state dependency inference.
    """

    def __init__(self, domain_name: str, package_name: Optional[str] = None, **_legacy):
        self.domain_name = domain_name
        self.package_name = package_name or domain_name
        # Adapter selection by implementation kind (composite probes source/stub/runtime)
        self.adapter = get_adapter_for_package(self.package_name)
        self.root_module = self.adapter.root_module
        self.discovered_modules: Dict[str, Any] = {}
        self.domain_types: Set[str] = set()
        self._source_cache: Dict[int, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Adapter-mediated evidence channels
    # ------------------------------------------------------------------
    def _source_of(self, obj: Any) -> Optional[str]:
        key = id(obj)
        if key not in self._source_cache:
            self._source_cache[key] = self.adapter.get_source(obj)
        return self._source_cache[key]

    def _doc_of(self, obj: Any) -> str:
        doc = self.adapter.get_docstring(obj)
        return (doc or "").strip()

    def discover_modules(self) -> Dict[str, Any]:
        self.discovered_modules = self.adapter.enumerate_modules(self.root_module)
        return self.discovered_modules

    def _discover_domain_type_universe(self) -> Set[str]:
        """
        Pass 1: Discover all domain-specific types T_domain without hardcoding.
        Collects types defined, aliased, or returned by the library that are not
        language-level primitives.
        """
        types: Set[str] = set()

        for m_name, m in self.discovered_modules.items():
            for attr_name in dir(m):
                if attr_name.startswith("_"):
                    continue
                try:
                    obj = _safe_getattr(m, attr_name)
                    if obj is None:
                        continue
                    # 1. Any class declared in the package
                    if inspect.isclass(obj):
                        cls_mod = getattr(obj, "__module__", "") or ""
                        if cls_mod.startswith(self.package_name) or cls_mod.startswith(f"_{self.package_name}"):
                            types.add(obj.__name__)
                    # 2. Any function return type
                    elif callable(obj):
                        sig = self.adapter.resolve_callable_signature(obj, callable_name=attr_name, mod=m)
                        if sig and sig.return_annotation is not inspect.Signature.empty:
                            ret_t = extract_clean_type_name(sig.return_annotation)
                            if ret_t and ret_t.lower() not in UNIVERSAL_BASE_PRIMITIVES:
                                types.add(ret_t)
                except Exception:
                    continue

        # Filter out universal primitives
        self.domain_types = {t for t in types if t.lower() not in UNIVERSAL_BASE_PRIMITIVES}
        return self.domain_types

    # ------------------------------------------------------------------
    # Stateful-lifecycle analysis (pure AST dataflow, zero name patterns)
    # ------------------------------------------------------------------
    def _analyze_class_lifecycle(
        self,
        cls: Any,
        method_sources: Dict[str, Optional[str]],
        mutator_names: Set[str],
        init_writes: Optional[Set[str]] = None,
    ) -> Dict[str, bool]:
        """
        Returns {method_name: is_state_dependent} for every non-mutator method
        of the class. A method is state-dependent iff it reads an instance
        attribute that some mutator writes but __init__ does NOT, OR it calls
        (via self) another state-dependent method (transitive closure over the
        intra-class call graph).

        The __init__ subtraction is the construction-availability rule: an
        attribute the initializer also writes exists from the moment the
        instance is constructed, so reading it cannot imply a lifecycle
        requirement (measured: pandas Series.to_csv reads self.name/self.index
        which in-place mutators may rewrite, but to_csv is valid on a freshly
        constructed Series). Only mutation-CREATED state counts. Pure static
        dataflow: no naming conventions, no library knowledge; libraries
        without source evidence yield no constraints (receiver stays "any").
        """
        state_dependent: Dict[str, bool] = {name: False for name in method_sources if name not in mutator_names}
        if not mutator_names:
            return state_dependent

        method_names = set(method_sources.keys())
        written: Set[str] = set()
        for m_name in mutator_names:
            src = method_sources.get(m_name)
            if src:
                written |= _self_attribute_writes(src, method_names)
        written -= set(init_writes or ())
        if not written:
            return state_dependent

        reads: Dict[str, Set[str]] = {}
        calls: Dict[str, Set[str]] = {}
        for m_name, src in method_sources.items():
            if m_name in mutator_names or not src:
                continue
            reads[m_name] = _self_attribute_reads(src, method_names)
            calls[m_name] = _self_method_calls(src)

        # Direct evidence: reads an attribute a mutator writes
        frontier: List[str] = []
        for m_name in state_dependent:
            if reads.get(m_name) and (reads[m_name] & written):
                state_dependent[m_name] = True
                frontier.append(m_name)

        # Transitive evidence: calls a state-dependent method via self
        while frontier:
            current = frontier.pop()
            for m_name, called in calls.items():
                if not state_dependent.get(m_name, False) and current in called:
                    state_dependent[m_name] = True
                    frontier.append(m_name)

        return state_dependent

    def _classify_mutator(
        self,
        ret_clean: str,
        ret_is_none: bool,
        fn: Any,
        cls_name: str,
        ret_is_union: bool = False,
    ) -> Tuple[bool, str]:
        """
        Mutator evidence:
          1. source: bare `return self` (definitive when source exists)
          2. -> None annotation (the in-place convention)
          3. -> Self annotation — ONLY when no source exists (stub-only
             libraries). Libraries WITH source may annotate `-> Self` on
             copy-returning methods (measured: pandas NDFrame.head is
             annotated Self yet returns self.iloc[:n], a fresh frame), so
             with source present, the source scan is the sole truth.
        A bare class-name return annotation is NOT evidence: copy-returning
        methods share it with fluent ones, and a union return
        (`DataFrame | None`) collapses to the class name while admitting a
        non-mutating path (measured: DataFrame.query silently became an
        endomorphism and dropped its filtered output).
        Returns (is_mutator, evidence_note).
        """
        src = self._source_of(fn)
        has_source = bool(src)
        if has_source and _returns_self_in_source(src):
            return True, "source:return self"
        if ret_clean and ret_clean.lower() not in _NONE_RETURNS and ret_clean.lower() not in _WILDCARD_RETURNS:
            # Concrete annotated return: only the stub-only Self channel can
            # still apply (no source to contradict the annotation).
            if not has_source and not ret_is_union and ret_clean.lower() == "self":
                return True, "annotation:Self (stub)"
            return False, ""
        if has_source and _returns_self_in_source(src):
            return True, "source:return self"
        if not has_source and not ret_is_union and ret_clean.lower() == "self":
            return True, "annotation:Self (stub)"
        if ret_is_none:
            return True, "annotation:None"
        return False, ""

    # ------------------------------------------------------------------
    # Main harvest pipeline
    # ------------------------------------------------------------------
    def harvest_all(self) -> List[CellSchema]:
        if not self.discovered_modules:
            self.discover_modules()
        if not self.domain_types:
            self._discover_domain_type_universe()

        cells: List[CellSchema] = []
        seen_ids: Set[str] = set()

        # 1. Harvest Constant Morphisms (Stage 1: 0 -> Const)
        for m_name, m in self.discovered_modules.items():
            for attr_name in dir(m):
                if attr_name.startswith("_"):
                    continue
                try:
                    val = _safe_getattr(m, attr_name)
                    if val is None or callable(val) or inspect.isclass(val) or inspect.ismodule(val):
                        continue
                    if not _is_constant_like(attr_name, val):
                        continue
                    cid = f"{self.domain_name.upper()}_{attr_name.upper()}"
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    tokens = sorted(list(CellTokenizer.tokenize_identifier(attr_name)))
                    cell = _harvest_constant_cell(
                        self.domain_name, cid, m_name,
                        code_expr=f"{m_name}.{attr_name}",
                        val=val, tokens=tokens,
                        dependencies=[f"import {m_name}"],
                    )
                    cells.append(cell)
                except Exception:
                    continue

        # 2. Harvest Classes: Constructors, Constants, and Methods
        for m_name, m in self.discovered_modules.items():
            for c_name in dir(m):
                if c_name.startswith("_"):
                    continue
                try:
                    cls = _safe_getattr(m, c_name)
                    if not cls or not inspect.isclass(cls):
                        continue
                    cls_mod = getattr(cls, "__module__", "") or ""
                    if not (cls_mod.startswith(self.package_name) or cls_mod.startswith(f"_{self.package_name}")):
                        continue

                    cells.extend(self._harvest_class(m_name, m, c_name, cls, cls_mod, seen_ids))
                except Exception:
                    continue

        # 3. Harvest Module-Level Functions
        for m_name, m in self.discovered_modules.items():
            for attr_name in dir(m):
                if attr_name.startswith("_"):
                    continue
                try:
                    fn = _safe_getattr(m, attr_name)
                    if not fn or not callable(fn) or inspect.isclass(fn):
                        continue
                    fn_mod = getattr(fn, "__module__", None) or m_name
                    if not (fn_mod.startswith(self.package_name) or fn_mod.startswith(f"_{self.package_name}")):
                        continue

                    cid = f"{self.domain_name.upper()}_{attr_name.upper()}"
                    if cid in seen_ids:
                        continue

                    sig = self.adapter.resolve_callable_signature(fn, callable_name=attr_name, mod=m)
                    if sig is None:
                        continue

                    seen_ids.add(cid)
                    doc = self._doc_of(fn)
                    first_doc = doc.splitlines()[0] if doc else f"{m_name}.{attr_name}"

                    inputs: Dict[str, PortSchema] = {}
                    required_template_args: List[str] = []
                    required_input_types: List[str] = []

                    for p in sig.parameters.values():
                        if p.kind is inspect.Parameter.VAR_POSITIONAL:
                            # Variadic data port: required generic carrier so the
                            # cell can actually receive data (e.g. *arrays).
                            inputs[p.name] = PortSchema(
                                type_name="any",
                                state="any",
                                required=True,
                                description=f"Variadic argument {p.name}"
                            )
                            required_template_args.append(f"{{{p.name}}}")
                            required_input_types.append("any")
                            continue
                        if p.kind is inspect.Parameter.VAR_KEYWORD:
                            continue
                        p_type = extract_clean_type_name(p.annotation)
                        is_req = (p.default is inspect.Parameter.empty)
                        inputs[p.name] = PortSchema(
                            type_name=p_type,
                            state="any",
                            required=is_req,
                            description=f"Argument {p.name}"
                        )
                        if is_req:
                            required_template_args.append(f"{{{p.name}}}")
                            required_input_types.append(p_type)

                    ret_type = extract_clean_type_name(sig.return_annotation)
                    ret_clean = ret_type if ret_type else "any"
                    ret_is_none = ret_clean.lower() in _NONE_RETURNS

                    # Pure Structural Monadic Profiling
                    consumes_domain = any(t in self.domain_types for t in required_input_types)
                    produces_domain = (ret_clean in self.domain_types)
                    primary_in_type = required_input_types[0] if required_input_types else "any"
                    prio = _module_priority(m_name)

                    if ret_is_none:
                        # In-place mutator (e.g. in-place scaling): the categorical
                        # output IS the mutated primary input — a monadic
                        # endomorphism D -> D — never a fabricated value.
                        stage = 2
                        node_type = "function"
                        node_role = "transform"
                        state = MUTATED_STATE
                        out_type = primary_in_type if primary_in_type not in ("", "any") else "any"
                        if required_template_args:
                            first_port = required_template_args[0][1:-1]
                            code_template = (
                                f"{m_name}.{attr_name}({', '.join(required_template_args)})\n"
                                f"{{output_var}} = {{{first_port}}}"
                            )
                        else:
                            code_template = f"{m_name}.{attr_name}()\n{{output_var}} = None"
                    elif consumes_domain and not produces_domain:
                        # D -> 1 (or D -> Base Primitives): SINK
                        stage = 3
                        node_type = "sink"
                        node_role = "sink"
                        state = "destination_written"
                        out_type = "str"
                        code_template = f"{m_name}.{attr_name}({', '.join(required_template_args)})\n{{output_var}} = 'done'"

                    elif not consumes_domain and produces_domain:
                        # 1 -> D (or Primitives -> D): SOURCE
                        stage = 1
                        node_type = "source"
                        node_role = "source"
                        state = "raw"
                        out_type = ret_clean
                        code_template = f"{{output_var}} = {m_name}.{attr_name}({', '.join(required_template_args)})"

                    elif consumes_domain and produces_domain:
                        # D1 -> D2: TRANSFORM / BRIDGE
                        stage = 2
                        if primary_in_type != "any" and primary_in_type != ret_clean:
                            node_type = "bridge"
                            node_role = "tunnel"
                            state = "raw"
                        else:
                            node_type = "function"
                            node_role = "transform"
                            state = "transformed"
                        out_type = ret_clean
                        code_template = f"{{output_var}} = {m_name}.{attr_name}({', '.join(required_template_args)})"

                    else:
                        # Primitives -> Primitives (or fallback utility)
                        stage = 2
                        node_type = "function"
                        node_role = "transform"
                        state = "transformed"
                        out_type = ret_clean if ret_clean != "any" else "any"
                        code_template = f"{{output_var}} = {m_name}.{attr_name}({', '.join(required_template_args)})"

                    if _is_fully_untyped(inputs, out_type):
                        prio = min(prio + 50, 100)

                    outputs = {
                        "output_data": PortSchema(
                            type_name=out_type,
                            state=state,
                            description=f"Output {state}"
                        )
                    }
                    tokens = sorted(list(CellTokenizer.tokenize_identifier(attr_name)))

                    cell = CellSchema(
                        cell_id=cid,
                        stage=stage,
                        inputs=inputs,
                        outputs=outputs,
                        code_template=code_template,
                        dependencies=[f"import {m_name}"],
                        semantic_tags=tokens,
                        keywords=tokens,
                        docstring=first_doc,
                        domain_name=self.domain_name,
                        node_type=node_type,
                        node_role=node_role,
                        source_priority=prio
                    )
                    cells.append(cell)
                except Exception:
                    continue

        return cells

    # ------------------------------------------------------------------
    # Per-class harvest: constructor, constants, instance/static/class methods
    # ------------------------------------------------------------------
    def _harvest_class(
        self,
        m_name: str,
        m: Any,
        c_name: str,
        cls: Any,
        cls_mod: str,
        seen_ids: Set[str],
    ) -> List[CellSchema]:
        cells: List[CellSchema] = []
        domain_prefix = self.domain_name.upper()

        # --- 2a. Constructor Introspection (Stage 2 zero-ary: () -> T_domain).
        # Constructors are NOT ingestion sources: they are composable
        # intermediate morphisms (estimator lifecycle: construct -> fit
        # -> predict/score), so they must be insertable mid-chain.
        ctor_sig = self.adapter.resolve_callable_signature(cls, callable_name="INIT", parent_cls_name=c_name, mod=m)
        if ctor_sig is not None:
            ctor_cid = f"{domain_prefix}_{c_name.upper()}_INIT"
            if ctor_cid not in seen_ids:
                seen_ids.add(ctor_cid)
                ctor_doc = self._doc_of(cls) or f"Construct a new {c_name}"

                ctor_inputs: Dict[str, PortSchema] = {}
                required_template_args: List[str] = []

                for p in ctor_sig.parameters.values():
                    if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                        continue
                    p_type = extract_clean_type_name(p.annotation)
                    is_req = (p.default is inspect.Parameter.empty)
                    ctor_inputs[p.name] = PortSchema(
                        type_name=p_type,
                        state="any",
                        required=is_req,
                        description=f"Constructor argument {p.name}"
                    )
                    if is_req:
                        required_template_args.append(f"{{{p.name}}}")

                ctor_tokens = sorted(list(CellTokenizer.tokenize_identifier(c_name)))
                ctor_expr = _public_alias(cls_mod, c_name)
                ctor_dep_mod = ctor_expr.rsplit(".", 1)[0]
                ctor_prio = _module_priority(ctor_dep_mod)
                ctor_cell = CellSchema(
                    cell_id=ctor_cid,
                    stage=2,
                    inputs=ctor_inputs,
                    outputs={
                        "output_data": PortSchema(
                            type_name=c_name,
                            state=CONSTRUCTED_STATE,
                            description=f"Newly constructed {c_name}"
                        )
                    },
                    code_template=f"{{output_var}} = {ctor_expr}({', '.join(required_template_args)})",
                    dependencies=[f"import {ctor_dep_mod}"],
                    semantic_tags=ctor_tokens,
                    keywords=ctor_tokens,
                    docstring=(ctor_doc.splitlines()[0] if ctor_doc else f"Construct {c_name}"),
                    domain_name=self.domain_name,
                    node_type="constructor",
                    node_role="constructor",
                    source_priority=ctor_prio
                )
                cells.append(ctor_cell)

        # --- 2b. Class-Scoped Constants and Enum Members
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(cls, attr_name, None)
                if val is None or callable(val) or inspect.isclass(val):
                    continue
                if not _is_constant_like(attr_name, val):
                    continue
                cattr_cid = f"{domain_prefix}_{c_name.upper()}_{attr_name.upper()}"
                if cattr_cid in seen_ids:
                    continue
                seen_ids.add(cattr_cid)
                cattr_tokens = sorted(list(
                    CellTokenizer.tokenize_identifier(attr_name)
                    | CellTokenizer.tokenize_identifier(c_name)
                ))
                cattr_cell = _harvest_constant_cell(
                    self.domain_name, cattr_cid, m_name,
                    code_expr=f"{cls_mod}.{c_name}.{attr_name}",
                    val=val, tokens=cattr_tokens,
                    dependencies=[f"import {cls_mod}"],
                )
                cells.append(cattr_cell)
            except Exception:
                continue

        # --- 2c. Instance / static / class methods — two-phase pipeline:
        #     Phase 1: introspect every method into raw records
        #     Phase 2: class-lifecycle stateflow analysis
        #     Phase 3: emit cells with evidence-based receiver states
        raw_attr = None
        method_records: List[Dict[str, Any]] = []

        for m_attr in dir(cls):
            if m_attr.startswith("_"):
                continue
            try:
                fn = getattr(cls, m_attr, None)
                if not callable(fn) or inspect.isclass(fn):
                    continue

                cid = f"{domain_prefix}_{c_name.upper()}_{m_attr.upper()}"
                if cid in seen_ids:
                    continue

                sig = self.adapter.resolve_callable_signature(fn, callable_name=m_attr, parent_cls_name=c_name, mod=m)
                if sig is None:
                    continue

                # Decorator kind via static inspection (structural, evidence-based)
                try:
                    static_attr = inspect.getattr_static(cls, m_attr)
                except Exception:
                    static_attr = None
                is_staticmethod = isinstance(static_attr, staticmethod)
                is_classmethod = isinstance(static_attr, classmethod)

                doc = self._doc_of(fn)
                first_doc = doc.splitlines()[0] if doc else f"{c_name}.{m_attr}"

                params = list(sig.parameters.values())
                if params and params[0].name in ("self", "cls"):
                    params = params[1:]
                elif is_classmethod and params:
                    params = params[1:]

                # Structural boilerplate filter
                if _is_boilerplate_config_method(params):
                    continue

                seen_ids.add(cid)

                ret_type = extract_clean_type_name(sig.return_annotation)
                ret_clean = ret_type if ret_type else "any"
                ret_is_none = ret_clean.lower() in _NONE_RETURNS

                method_records.append({
                    "cid": cid,
                    "attr": m_attr,
                    "fn": fn,
                    "sig": sig,
                    "params": params,
                    "doc": first_doc,
                    "ret_clean": ret_clean,
                    "ret_is_none": ret_is_none,
                    "is_static": is_staticmethod,
                    "is_class": is_classmethod,
                })
            except Exception:
                continue

        # Phase 2: mutator detection + receiver-state dependency analysis.
        # Bound instance methods only (static/class methods have no receiver).
        # The ANALYSIS set includes private helper methods (single-underscore,
        # not harvested as cells): state evidence often lives behind them
        # (e.g. predict -> _transform -> reads mutator-written attributes),
        # and the transitive closure over the intra-class call graph needs
        # those nodes to propagate constraints to the public surface.
        instance_records = [r for r in method_records if not (r["is_static"] or r["is_class"])]

        def _raw_ret_is_union(fn_obj: Any) -> bool:
            """True iff the declared return is a union/optional — never trusted
            as a bare class return for mutator evidence."""
            try:
                raw = getattr(fn_obj, "__annotations__", {}).get("return", "")
                s = str(raw)
                return "|" in s or "Optional" in s or "Union" in s
            except Exception:
                return False

        analysis_sources: Dict[str, Optional[str]] = {}
        analysis_mutators: Set[str] = set()
        for attr_name in dir(cls):
            if attr_name.startswith("__"):
                continue  # dunders are protocol hooks, not lifecycle state
            try:
                static_attr = inspect.getattr_static(cls, attr_name)
                if isinstance(static_attr, (staticmethod, classmethod, property)):
                    continue
                member_fn = getattr(cls, attr_name, None)
                if not callable(member_fn) or inspect.isclass(member_fn):
                    continue
            except Exception:
                continue
            analysis_sources[attr_name] = self._source_of(member_fn)
            # Mutator classification for the analysis set: source convention
            # first (return self); `-> Self` annotations count ONLY when no
            # source exists (stub-only) because source-owning libraries may
            # annotate Self on copy-returning methods. Union returns are
            # never annotation evidence (see _classify_mutator).
            if _returns_self_in_source(analysis_sources[attr_name]):
                analysis_mutators.add(attr_name)
                continue
            if analysis_sources[attr_name] is None:
                # Stub-only: annotation is the only available channel
                if not _raw_ret_is_union(member_fn):
                    try:
                        _sig = self.adapter.resolve_callable_signature(member_fn, callable_name=attr_name, parent_cls_name=c_name, mod=m)
                        if _sig is not None:
                            _ret = extract_clean_type_name(_sig.return_annotation)
                            if _ret.lower() in ("none", "nonetype", "void", "noreturn", "self"):
                                analysis_mutators.add(attr_name)
                    except Exception:
                        pass
                continue
            # Source available: only the None annotation adds evidence beyond
            # the source scan (in-place convention).
            if not _raw_ret_is_union(member_fn):
                try:
                    _sig = self.adapter.resolve_callable_signature(member_fn, callable_name=attr_name, parent_cls_name=c_name, mod=m)
                    if _sig is not None:
                        _ret = extract_clean_type_name(_sig.return_annotation)
                        if _ret.lower() in ("none", "nonetype", "void", "noreturn"):
                            analysis_mutators.add(attr_name)
                except Exception:
                    pass

        # Construction-availability rule: attributes the initializer chain
        # (cls.__mro__ __init__s — pandas Series.__init__ delegates to
        # NDFrame.__init__ which writes _mgr) writes exist from construction;
        # reading them carries no lifecycle constraint.
        try:
            all_names = set(analysis_sources.keys())
            init_writes: Set[str] = set()
            for base in getattr(cls, "__mro__", (cls,)):
                try:
                    base_init = getattr(base, "__init__", None)
                    base_src = self._source_of(base_init) if base_init is not None else None
                    if base_src:
                        init_writes |= _self_attribute_writes(base_src, all_names)
                except Exception:
                    continue
        except Exception:
            init_writes = set()

        state_dependent = self._analyze_class_lifecycle(
            cls, analysis_sources, analysis_mutators, init_writes
        )

        # Phase 3: emit method cells
        for rec in method_records:
            try:
                cells.extend(self._emit_method_cell(m_name, c_name, cls_mod, rec, analysis_mutators, state_dependent))
            except Exception:
                continue

        return cells

    def _emit_method_cell(
        self,
        m_name: str,
        c_name: str,
        cls_mod: str,
        rec: Dict[str, Any],
        mutator_names: Set[str],
        state_dependent: Dict[str, bool],
    ) -> List[CellSchema]:
        attr = rec["attr"]
        sig = rec["sig"]
        params: List[inspect.Parameter] = rec["params"]
        ret_clean: str = rec["ret_clean"]
        is_static: bool = rec["is_static"]
        is_class: bool = rec["is_class"]
        cid: str = rec["cid"]

        required_template_args: List[str] = []
        inputs: Dict[str, PortSchema] = {}

        receiver_state = "any"
        is_mutator = (not is_static and not is_class and attr in mutator_names)

        if is_class:
            # Classmethod: an ALTERNATIVE CONSTRUCTOR only when its declared
            # return is the class itself (or unannotated) — e.g. pd.Timestamp
            # .fromisoformat. A classmethod returning anything else (a
            # predicate like is_dtype -> bool, a lookup returning str) is a
            # plain producer, never a fabricated constructor.
            for p in params:
                if p.kind is inspect.Parameter.VAR_POSITIONAL:
                    inputs[p.name] = PortSchema(type_name="any", state="any", required=True,
                                                description=f"Variadic argument {p.name}")
                    required_template_args.append(f"{{{p.name}}}")
                    continue
                if p.kind is inspect.Parameter.VAR_KEYWORD:
                    continue
                p_type = extract_clean_type_name(p.annotation)
                is_req = (p.default is inspect.Parameter.empty)
                inputs[p.name] = PortSchema(type_name=p_type, state="any", required=is_req,
                                            description=f"Argument {p.name}")
                if is_req:
                    required_template_args.append(f"{{{p.name}}}")

            cls_expr = _public_alias(cls_mod, c_name)
            ret_l = ret_clean.lower()
            is_alt_ctor = ret_l in ("", "any", "*", "unknown", c_name.lower(), "self")
            if is_alt_ctor:
                out_state = CONSTRUCTED_STATE
                out_type = c_name
                stage = 2
                node_type = "constructor"
                node_role = "constructor"
            else:
                stage = 2
                node_type = "bridge"
                node_role = "tunnel"
                out_state = "raw"
                out_type = ret_clean
            code_template = f"{{output_var}} = {cls_expr}.{attr}({', '.join(required_template_args)})"
            prio = _module_priority(cls_mod.split("._", 1)[0])

        elif is_static:
            # Staticmethod: a plain function that happens to live on the class —
            # NO receiver port (fabricating one would demand an instance that
            # the semantics never needed).
            for p in params:
                if p.kind is inspect.Parameter.VAR_POSITIONAL:
                    inputs[p.name] = PortSchema(type_name="any", state="any", required=True,
                                                description=f"Variadic argument {p.name}")
                    required_template_args.append(f"{{{p.name}}}")
                    continue
                if p.kind is inspect.Parameter.VAR_KEYWORD:
                    continue
                p_type = extract_clean_type_name(p.annotation)
                is_req = (p.default is inspect.Parameter.empty)
                inputs[p.name] = PortSchema(type_name=p_type, state="any", required=is_req,
                                            description=f"Argument {p.name}")
                if is_req:
                    required_template_args.append(f"{{{p.name}}}")

            cls_expr = _public_alias(cls_mod, c_name)
            consumes_domain = any(
                extract_clean_type_name(p.annotation) in self.domain_types
                for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
            produces_domain = ret_clean in self.domain_types
            if ret_clean.lower() in _NONE_RETURNS:
                stage, node_type, node_role = 2, "function", "transform"
                out_state = MUTATED_STATE
                out_type = "any"
                code_template = f"{cls_expr}.{attr}({', '.join(required_template_args)})\n{{output_var}} = None"
            elif produces_domain:
                stage, node_type, node_role = 2, "function", "transform"
                out_state = "raw"
                out_type = ret_clean
                code_template = f"{{output_var}} = {cls_expr}.{attr}({', '.join(required_template_args)})"
            elif consumes_domain:
                stage, node_type, node_role = 3, "sink", "sink"
                out_state = "destination_written"
                out_type = "str"
                code_template = f"{cls_expr}.{attr}({', '.join(required_template_args)})\n{{output_var}} = 'done'"
            else:
                stage, node_type, node_role = 2, "bridge", "tunnel"
                out_state = "raw"
                out_type = ret_clean if ret_clean != "any" else "any"
                code_template = f"{{output_var}} = {cls_expr}.{attr}({', '.join(required_template_args)})"
            prio = _module_priority(cls_mod.split("._", 1)[0])

        else:
            # Bound instance method: receiver port + arguments. The receiver's
            # declared state is EVIDENCE-BASED: mutators accept any lifecycle
            # stage and transition it to "mutated"; methods that read
            # mutator-written state (or call such methods) REQUIRE C[mutated];
            # everything else keeps the permissive "any" (no evidence -> no
            # fabricated constraint).
            if is_mutator:
                receiver_state = "any"
            else:
                receiver_state = MUTATED_STATE if state_dependent.get(attr, False) else "any"

            inputs["data"] = PortSchema(
                type_name=c_name,
                state=receiver_state,
                required=True,
                description=f"Receiver instance of {c_name}"
                + (f" (requires mutated state via {attr})" if receiver_state == MUTATED_STATE else "")
            )

            for p in params:
                if p.kind is inspect.Parameter.VAR_POSITIONAL:
                    inputs[p.name] = PortSchema(type_name="any", state="any", required=True,
                                                description=f"Variadic argument {p.name}")
                    required_template_args.append(f"{{{p.name}}}")
                    continue
                if p.kind is inspect.Parameter.VAR_KEYWORD:
                    continue
                p_type = extract_clean_type_name(p.annotation)
                is_req = (p.default is inspect.Parameter.empty)
                inputs[p.name] = PortSchema(type_name=p_type, state="any", required=is_req,
                                            description=f"Argument {p.name}")
                if is_req:
                    required_template_args.append(f"{{{p.name}}}")

            prio = _module_priority(cls_mod.split("._", 1)[0])
            produces_domain = (ret_clean in self.domain_types) or (ret_clean.lower() == c_name.lower())

            if is_mutator:
                # Monadic endomorphism on the receiver's own type:
                # C[any] -> C[mutated]. The mutated instance stays composable
                # (construct -> fit -> predict -> score) and the typestate
                # algebra now CONNECTS fit to its state-dependent consumers.
                stage = 2
                node_type = "function"
                node_role = "transform"
                out_state = MUTATED_STATE
                out_type = c_name
                code_template = f"{{data}}.{attr}({', '.join(required_template_args)})\n{{output_var}} = {{data}}"
            elif produces_domain:
                stage = 2
                if ret_clean.lower() != c_name.lower():
                    node_type = "bridge"
                    node_role = "tunnel"
                    out_state = "raw"
                    out_type = ret_clean
                else:
                    node_type = "function"
                    node_role = "transform"
                    out_state = "transformed"
                    out_type = c_name
                code_template = f"{{output_var}} = {{data}}.{attr}({', '.join(required_template_args)})"
            else:
                # Unknown / primitive return: a composable stage-2
                # producer (e.g. score -> float). Methods are never
                # terminal egress; egress is a written-artifact port.
                stage = 2
                node_type = "bridge"
                node_role = "tunnel"
                out_state = "raw"
                out_type = ret_clean if ret_clean != "any" else "any"
                code_template = f"{{output_var}} = {{data}}.{attr}({', '.join(required_template_args)})"

        if _is_fully_untyped(inputs, out_type):
            prio = min(prio + 50, 100)

        outputs = {
            "output_data": PortSchema(
                type_name=out_type,
                state=out_state,
                description=f"Output {out_state}"
            )
        }
        tokens = sorted(list(
            CellTokenizer.tokenize_identifier(attr)
            | CellTokenizer.tokenize_identifier(c_name)
        ))
        method_dep_mod = cls_mod.split("._", 1)[0] if "._" in cls_mod else cls_mod

        cell = CellSchema(
            cell_id=cid,
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            code_template=code_template,
            dependencies=[f"import {method_dep_mod}"],
            semantic_tags=tokens,
            keywords=tokens,
            docstring=rec["doc"],
            domain_name=self.domain_name,
            node_type=node_type,
            node_role=node_role,
            source_priority=prio
        )
        return [cell]

    # ------------------------------------------------------------------
    # Knowledge merge (old tree knowledge onto fresh cells)
    # ------------------------------------------------------------------
    @classmethod
    def enrich_from_existing_trees(
        cls,
        domain_name: str,
        new_cells: List[CellSchema],
        existing_tree_paths: List[Union[str, Path]]
    ) -> List[CellSchema]:
        old_knowledge: Dict[str, Dict[str, Any]] = {}
        for path in existing_tree_paths:
            p = Path(path)
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                old_cells = data if isinstance(data, list) else data.get("cells", [])
                for oc in old_cells:
                    if isinstance(oc, dict) and "cell_id" in oc:
                        cid = oc["cell_id"].upper()
                        old_knowledge[cid] = oc
                        old_knowledge[cid.replace("_", "")] = oc
            except Exception:
                pass

        for cell in new_cells:
            cid = cell.cell_id.upper()
            suffix = ("_" + cid.split("_", 1)[1]) if "_" in cid else cid
            match = (
                old_knowledge.get(cid)
                or old_knowledge.get(cid.replace("_", ""))
                or next((v for k, v in old_knowledge.items() if k.endswith(suffix)), None)
            )
            if match:
                old_prio = match.get("source_priority", 100)
                if old_prio <= 10:
                    cell.source_priority = old_prio
                old_tags = match.get("semantic_tags", [])
                if old_tags:
                    cell.semantic_tags = list(dict.fromkeys(cell.semantic_tags + old_tags))
                old_kws = match.get("keywords", [])
                if old_kws:
                    cell.keywords = list(dict.fromkeys(cell.keywords + old_kws))
                old_doc = match.get("docstring", "")
                if old_doc and len(old_doc) > len(cell.docstring):
                    cell.docstring = old_doc
                if match.get("verified", False):
                    cell.verified = True
        return new_cells

    def harvest_and_save(self, out_file: Union[str, Path]) -> TreeSchema:
        cells = self.harvest_all()
        out_path = Path(out_file)
        if out_path.exists():
            cells = self.enrich_from_existing_trees(self.domain_name, cells, [out_path])
        tree = TreeSchema(domain=self.domain_name, cells=cells)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tree.model_dump_json(indent=2))
        logger.info(f"[{self.domain_name}] Successfully harvested {len(cells)} cells to {out_path}")
        return tree


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 universal_harvester.py <library_name>")
        sys.exit(1)
    lib = sys.argv[1]
    harvester = UniversalHarvester(lib)
    out_target = Path("trees") / f"{lib}.json"
    tree = harvester.harvest_and_save(out_target)
    by_archetype: Dict[str, int] = {}
    for c in tree.cells:
        k = f"Stage {c.stage} | Type: {c.node_type} | Role: {c.node_role}"
        by_archetype[k] = by_archetype.get(k, 0) + 1
    print(f"\n[NSTL Universal Harvester] Completed harvest of '{lib}':")
    print(f"Adapter chain: {harvester.adapter.describe()}")
    print(f"Discovered Domain Types: {sorted(list(harvester.domain_types))[:10]}... ({len(harvester.domain_types)} total)")
    print(f"Total cells generated: {len(tree.cells)}")
    print("\nArchetype breakdown:")
    for k, count in sorted(by_archetype.items()):
        print(f"  {k}: {count} nodes")
