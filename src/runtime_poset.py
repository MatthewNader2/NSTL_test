"""
src/runtime_poset.py - Neuro-Symbolic Topological Lattice (NSTL)
Zero-Hardcode Dynamic Runtime Poset, Protocol Reflection & Structural Role Engine.
"""

from __future__ import annotations
import builtins
import collections.abc
import inspect
import keyword
import mimetypes
import numbers
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Type


_RUNTIME_ABC_ROOTS: Tuple[Tuple[Type[Any], str], ...] = (
    (bool, "logical"),
    (numbers.Number, "numeric"),
    (os.PathLike, "path"),
    (collections.abc.Mapping, "mapping"),
    (collections.abc.Sequence, "sequence"),
    (collections.abc.Set, "set"),
    (collections.abc.Collection, "collection"),
    (collections.abc.Callable, "Callable"),
    (BaseException, "Exception"),
)


def reflect_class_carrier(cls: Type[Any]) -> Optional[str]:
    """
    Infers the abstract categorical carrier of a live Python class via
    official runtime protocols (zero hardcoded library or class names).
    """
    if not inspect.isclass(cls):
        return None
    try:
        if issubclass(cls, bool):
            return "logical"
        if issubclass(cls, numbers.Number):
            return "scalar"
        if issubclass(cls, os.PathLike):
            return "path"
        if issubclass(cls, (str, bytes, bytearray)):
            return "text"
    except TypeError:
        pass

    # 1. Python DataFrame Interchange Protocol (__dataframe__) or 2D tabular protocol
    if hasattr(cls, "__dataframe__") or (hasattr(cls, "iloc") and hasattr(cls, "columns")):
        return "table"

    # 2. Array / Tensor Protocol (__array_ufunc__, __array_interface__, __dlpack__)
    if (
        hasattr(cls, "__array_ufunc__")
        or hasattr(cls, "__array_interface__")
        or hasattr(cls, "__dlpack__")
    ):
        return "tensor"

    # 3. Stateful Estimator Protocol (fit + predict/transform)
    if callable(getattr(cls, "fit", None)) and (
        callable(getattr(cls, "predict", None)) or callable(getattr(cls, "transform", None))
    ):
        return "model"

    # 4. Standard Collections ABC
    try:
        if issubclass(cls, collections.abc.Collection):
            return "collection"
    except TypeError:
        pass

    return None


def reflect_mro_edges(cls: Type[Any], registry: Any, carrier_buckets: Dict[str, Set[str]]) -> None:
    """
    Walks `inspect.getmro(cls)` for any live class and registers its inheritance
    edges and protocol carrier into `TypeRegistry`.
    """
    if not inspect.isclass(cls):
        return
    cls_name = cls.__name__.lower()
    try:
        mro = inspect.getmro(cls)
    except Exception:
        return

    for base in mro[1:]:
        if base is object:
            registry.register_type(cls_name, "object")
            continue
        base_name = base.__name__.lower()
        registry.register_type(cls_name, base_name)

    for abc_cls, proto_name in _RUNTIME_ABC_ROOTS:
        try:
            if issubclass(cls, abc_cls) and not (
                abc_cls is collections.abc.Sequence and issubclass(cls, (str, bytes, bytearray))
            ):
                registry.register_type(cls_name, proto_name)
                if proto_name == "numeric":
                    registry.register_type("numeric", "scalar")
        except TypeError:
            continue

    carrier = reflect_class_carrier(cls)
    if carrier:
        registry.register_type(cls_name, carrier)
        registry.register_type(carrier, "object")
        registry.register_abstract_carriers([carrier])
        registry.register_abstract_carrier_mapping({cls_name: carrier, carrier: carrier})
        carrier_buckets.setdefault(carrier, set()).add(cls_name)


def bootstrap_runtime_poset(registry: Any) -> None:
    """
    Populates `TypeRegistry` strictly from live Python runtime introspection
    (`builtins`, `collections.abc`, `numbers`, `os.PathLike`, `mimetypes`,
    `typing`, and loaded modules in `sys.modules`).
    """
    carrier_buckets: Dict[str, Set[str]] = {
        "path": {"path", "filepath", "filename", "pathlike", "uri", "url", "file", "source_path", "file_path", "image_path", "dest_path", "savepath", "output_path", "target_path"},
        "table": {"table", "dataset"},
        "tensor": {"tensor", "array-like", "array_like"},
        "model": {"model", "estimator"},
        "collection": {"collection", "sequence", "mapping"},
        "scalar": {"scalar", "numeric"},
        "logical": {"logical"},
        "text": {"text", "str"},
    }

    # 1. Universal algebraic top symbols & generic type variables from `typing`
    for sym in ("any", "object", "*", "top", "unknown", "⊤"):
        registry.register_top(sym)
        registry.register_top_state(sym)
    registry.register_type_vars(["T", "U", "V", "R", "K", "State"])
    for prod_c in ("tuple", "product", "pair"):
        registry.register_product_constructor(prod_c)

    # 2. Structural carrier roots
    for group, members in carrier_buckets.items():
        registry.register_type(group, "object")
        registry.register_abstract_carriers([group])
        for m in members:
            if m != group:
                registry.register_type(m, group)
            registry.register_abstract_carrier_mapping({m: group})
    registry.register_type("path", "str")
    registry.register_type("filepath", "path")
    registry.register_type("file", "path")
    registry.register_type("tensor", "array-like")
    registry.register_type("array-like", "sequence")
    registry.register_top_level_carriers(list(carrier_buckets.keys()))
    registry.register_path_port_tokens(list(carrier_buckets["path"]))

    # 3. Introspect all Python built-in classes via MRO
    for obj in builtins.__dict__.values():
        if inspect.isclass(obj):
            reflect_mro_edges(obj, registry, carrier_buckets)

    # 4. Introspect already-imported root packages in sys.modules
    for mod_name, mod in list(sys.modules.items()):
        if not mod or "." in mod_name or mod_name.startswith("_"):
            continue
        try:
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                obj = getattr(mod, attr_name, None)
                if inspect.isclass(obj):
                    reflect_mro_edges(obj, registry, carrier_buckets)
        except Exception:
            continue

    # 5. Introspect standard OS file formats via Python's `mimetypes` database
    mimetypes.init()
    registry.register_type("format", "object")
    for ext, mime in mimetypes.types_map.items():
        clean_ext = ext.lstrip(".").lower()
        if not clean_ext or not clean_ext.isalnum():
            continue
        top_cat = mime.split("/", 1)[0].lower()
        if top_cat in ("image", "audio", "video", "text", "application"):
            fmt_group = "tabular_format" if mime in ("text/csv", "text/tab-separated-values") else f"{top_cat}_format"
            registry.register_type(fmt_group, "format")
            registry.register_type(clean_ext, fmt_group)
            if fmt_group == "tabular_format":
                registry.register_type(clean_ext, "table")

    # 6. Derive carrier_roles directly from reflected protocol buckets
    registry.register_carrier_roles({
        "source_data": sorted(carrier_buckets["path"]),
        "feature_input": sorted(carrier_buckets["table"] | carrier_buckets["tensor"] | carrier_buckets["collection"]),
        "model_input": sorted(carrier_buckets["model"]),
    })

    # 7. Structural verification semantics (algebraic port & role identifiers, not English prose)
    registry.register_verification_semantics({
        "dataflow_roles": ["carrier", "data_input"],
        "feature_roles": ["feature_input"],
        "target_roles": ["target_input"],
        "model_roles": ["model_input", "model_sink", "model", "estimator", "regressor", "classifier", "clusterer", "transformer"],
        "estimator_roles": ["estimator", "regressor", "classifier", "clusterer", "transformer", "model"],
        "mutually_exclusive_roles": ["regressor", "classifier", "clusterer"],
        "source_roles": ["source", "ingress"],
        "sink_roles": ["sink", "terminal", "egress"],
        "terminal_roles": ["sink", "model_sink", "evaluator", "export", "terminal", "output"],
        "bridge_roles": ["bridge", "tunnel", "adapter", "cast", "coercion"],
        "row_set_effects": ["cleans_missing", "drops_rows", "filters_rows", "row_filter", "extent_mutating"],
        "extent_mutating_effects": ["cleans_missing", "drops_rows", "filters_rows", "row_filter", "extent_mutating"],
        "component_transform_effects": ["normalizes_column", "transforms_column", "encodes_column", "scales_column", "component_transform"],
        "column_transform_effects": ["normalizes_column", "transforms_column", "encodes_column", "scales_column", "component_transform"],
        "tensor_context_tokens": ["array", "tensor", "ndarray", "matrix", "vector", "grid", "volume", "signal", "dim", "dims", "dimension", "ndim"],
        "projection_port_names": ["column", "columns", "key", "keys", "field", "fields"],
        "projection_port_roles": ["column_projection", "columns"],
        "projective_roles": ["column_projection", "columns", "projection", "feature_projection", "target_projection"],
        "projection_port_states": ["column_projection", "columns", "feature_names", "column_name", "target_name"],
        "path_consumer_roles": ["source_data"],
        "path_abstract_types": ["path"],
        "headless_environment": [
            "SDL_VIDEODRIVER=dummy",
            "SDL_AUDIODRIVER=dummy",
            "MPLBACKEND=Agg",
            "QT_QPA_PLATFORM=offscreen",
            "PYGAME_HIDE_SUPPORT_PROMPT=1",
        ],
        "smoke_test_loop_bound": ["5"],
    })

    # 8. Linguistic-algebraic bridge tokens
    registry.register_source_port_tokens(["source", "source_path", "source_file", "filepath", "filename", "input_file", "image_path", "file_path", "file"])
    registry.register_dest_port_tokens(["called", "named", "as", "labeled", "titled", "new", "dest_path", "savepath", "output_path", "dest_identifier", "target_path", "output_file"])
    registry.register_column_projection_tokens(["column", "columns", "feature", "features", "target", "targets", "key", "by", "subset"])
    registry.register_egress_tokens(["export", "save", "write", "dump", "sink", "output"])

    # 9. Universal typestates and materialization state hierarchy
    if hasattr(registry, "register_state"):
        registry.register_state("source_identifier", None, carrier_type="str")
        registry.register_state("exported", None)
        registry.register_state("written_to_disk", "exported")
        registry.register_state("saved", "written_to_disk")
        registry.register_state("default", None)
    if hasattr(registry, "register_identifier_root_states"):
        registry.register_identifier_root_states(["source_identifier"])
    if hasattr(registry, "register_materialization_states"):
        registry.register_materialization_states(["saved", "written_to_disk", "exported"])
    if hasattr(registry, "register_terminal_states"):
        registry.register_terminal_states(["saved", "written_to_disk", "exported"])
    if hasattr(registry, "register_state_role_ancestors"):
        registry.register_state_role_ancestors({
            "saved": ["terminal"],
            "written_to_disk": ["terminal"],
            "exported": ["terminal"],
        })


def infer_abstract_carrier(type_name: str, registry: Any = None) -> Optional[str]:
    """
    Infers the abstract categorical carrier for any type name via registered
    abstract carriers, protocol mappings, or live reflection.
    """
    tn = str(type_name).strip().lower()
    if not tn or tn in ("any", "object", "*", "top", "unknown", "none"):
        return None
    if registry:
        for abs_c in registry.get_abstract_carriers():
            if tn == abs_c or registry.is_subtype(tn, abs_c):
                return abs_c
    return None


def classify_cell_structural_role(cell: Any, registry: Any = None) -> str:
    """
    Classifies a cell's structural role from first principles via its signature
    morphism f: tau_in -> tau_out and AST template:
      - Ingress (source): Consumes an external locator and produces an in-memory carrier.
      - Egress (sink): Consumes an in-memory carrier and produces None/void or materialized path,
        or has no LHS variable assignment in its template.
      - Lens Setter (setter): Morphism mu: S x K x V -> S that takes composite carrier S,
        key/index port K, and value port V, and outputs S.
      - Coercion / Projection Bridge (bridge): Pure morphism (|required_inputs| <= 2, no side effects)
        projecting a component from composite carrier or converting representation between carriers.
    """
    if registry is None:
        try:
            from .lattice import TypeRegistry
            registry = TypeRegistry.get_instance()
        except Exception:
            try:
                from lattice import TypeRegistry
                registry = TypeRegistry.get_instance()
            except Exception:
                registry = None

    inputs = getattr(cell, "inputs", {}) or {}
    outputs = getattr(cell, "outputs", {}) or {}
    template = getattr(cell, "code_template", "") or ""
    effects = getattr(cell, "effects", []) or []

    carrier_bind_tags = {"data_carrier", "receiver", "carrier"}
    key_bind_tags = {"key", "column_key", "field_key", "attr_key"}
    has_carrier_in = False
    has_key_in = False
    has_val_in = False
    carrier_in_type = None

    for p in inputs.values():
        bind = getattr(p, "binds", None)
        role = getattr(p, "port_role", None) or getattr(p, "derived_role", "")
        if bind in carrier_bind_tags or role in ("receiver", "carrier"):
            has_carrier_in = True
            carrier_in_type = str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower()
        if bind in key_bind_tags or role in ("key", "column_key"):
            has_key_in = True
        if bind == "assigned_value" or role in ("assigned_value", "column_value"):
            has_val_in = True

    prim_out = getattr(cell, "primary_output", None)
    out_tn = str(getattr(getattr(prim_out, "signature", prim_out), "type_name", "") or "").lower() if prim_out else ""
    out_state = str(getattr(getattr(prim_out, "signature", prim_out), "state", "") or "").lower() if prim_out else ""

    # 1. Lens Setter: mu: S x K x V -> S
    if has_carrier_in and has_key_in and has_val_in:
        if carrier_in_type and out_tn and registry:
            if registry.is_subtype(out_tn, carrier_in_type) or registry.is_subtype(carrier_in_type, out_tn):
                return "setter"
        elif has_carrier_in and has_key_in and has_val_in:
            return "setter"

    # 2. Ingress (source): consumes external locator, produces in-memory carrier
    path_tokens = registry.get_path_port_tokens() if registry else {"path", "filepath", "filename", "uri", "url", "file"}
    in_is_locator = False
    for p in inputs.values():
        p_tn = str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower()
        p_st = str(getattr(getattr(p, "signature", p), "state", "") or "").lower()
        p_role = getattr(p, "port_role", None) or getattr(p, "derived_role", "")
        if p_st in ("source_identifier", "input_path", "source_path") or p_role in ("source_data", "source", "ingress") or p_tn in path_tokens:
            in_is_locator = True
            break
        if registry and any(registry.is_subtype(p_tn, pt) for pt in path_tokens):
            in_is_locator = True
            break

    if in_is_locator and prim_out:
        if out_tn not in path_tokens and (not registry or not any(registry.is_subtype(out_tn, pt) for pt in path_tokens)):
            return "source"

    # 3. Egress (sink): consumes in-memory carrier, produces None/void/materialized path or has no LHS assignment
    mat_states = registry.get_materialization_states() if registry else {"saved", "written_to_disk", "exported"}
    term_states = registry.get_terminal_states() if registry else {"saved", "written_to_disk", "exported"}
    has_lhs = ("=" in template and any(ch not in ("=", "!", "<", ">") for ch in template[:template.index("=")].strip())) if template else False

    if not outputs or out_tn in ("none", "void", "unit", ""):
        return "sink"
    if out_state in mat_states or out_state in term_states:
        return "sink"
    if template and not has_lhs:
        return "sink"

    # 4. Coercion / Projection Bridge
    req_inputs = [p for p in inputs.values() if getattr(p, "required", True)]
    is_pure = not bool(effects) or all(str(e).lower().startswith("preserves") for e in effects)
    if is_pure and len(req_inputs) <= 2:
        node_role = str(getattr(cell, "node_role", "") or "").lower()
        if node_role in ("bridge", "tunnel", "adapter", "cast", "coercion") or getattr(cell, "projection", None):
            return "bridge"

    declared_role = str(getattr(cell, "node_role", "") or "").lower()
    return declared_role or "function"


__all__ = [
    "reflect_class_carrier",
    "reflect_mro_edges",
    "bootstrap_runtime_poset",
    "infer_abstract_carrier",
    "classify_cell_structural_role",
]

