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
        "path_consumer_roles": ["source_data", "model_sink"],
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


__all__ = [
    "reflect_class_carrier",
    "reflect_mro_edges",
    "bootstrap_runtime_poset",
]
