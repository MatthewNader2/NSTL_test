"""
src/library_adapters.py - Neuro-Symbolic Topological Lattice (NSTL)
Adaptive evidence-source adapters for the Unified Harvest Pipeline.

Architecture (one pipeline, adaptive adapters):

  ┌──────────────────────────── Unified Harvest Pipeline ───────────────────────────┐
  │  constants → constructors → instance methods → module functions → typestate     │
  │  (the pipeline asks the adapter for EVIDENCE; it owns all schema semantics)     │
  └──────────▲──────────────▲──────────────▲──────────────▲───────────────────────┘
             │ evidence     │ evidence     │ evidence     │ evidence
   ┌─────────┴──────┐ ┌─────┴─────────┐ ┌──┴──────────┐ ┌─┴───────────────┐
   │ PurePythonSrc  │ │ TypingStub    │ │ Runtime     │ │  ...future      │
   │ Adapter        │ │ Adapter       │ │ Reflection  │ │  adapters       │
   └────────────────┘ └───────────────┘ └─────────────┘ └─────────────────┘

An adapter is selected BY THE IMPLEMENTATION KIND OF THE LIBRARY, never by its
name: what matters is which evidence channels a library can physically provide
(Python source? typing stubs? runtime reflection only?). Libraries built the
same way are served by the same adapter. The CompositeLibraryAdapter probes the
root package and chains adapters from richest to poorest evidence, so hybrid
libraries (pure-Python surface over C internals) get every channel they have.

Zero domain hardcodes: no library names, no library-specific heuristics.
"""

from __future__ import annotations

import abc
import functools
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from log_config import get_logger
    from signature_introspector import (
        resolve_signature,
        _find_stub_file_for_module,
    )
except ImportError:
    from .log_config import get_logger
    from .signature_introspector import (
        resolve_signature,
        _find_stub_file_for_module,
    )

logger = get_logger("library_adapters")


# =====================================================================
# 1. Evidence requests (what the pipeline asks adapters to provide)
# =====================================================================

class Evidence:
    """
    Names the evidence channels the unified pipeline consumes.
    Adapters answer a request or decline (return None) — the composite then
    falls through to the next adapter in richness order.
    """
    MODULES = "modules"          # module enumeration: Dict[module_name, module]
    SOURCE = "source"            # Python source text (enables AST dataflow evidence)
    DOCSTRING = "docstring"      # documentation text
    SIGNATURE = "signature"      # inspect.Signature for a callable


def _is_python_source_module(module: Any) -> bool:
    """True iff the module is backed by real Python source (not a C extension)."""
    file_path = getattr(module, "__file__", None) or ""
    if not file_path:
        return False
    # Extension modules are .so / .pyd; source modules are .py
    return str(file_path).endswith(".py")


def _walk_package(root_module: Any) -> Dict[str, Any]:
    """
    Universal module enumeration for an importable package (or single module).
    Excludes test suites and private subpackages — structural, name-agnostic
    except for the universal Python test-convention markers.
    """
    package_name = getattr(root_module, "__name__", "")
    modules: Dict[str, Any] = {package_name: root_module}
    pkg_path = getattr(root_module, "__path__", None)

    def _excluded(name: str) -> bool:
        parts = name.split(".")
        for p in parts:
            pl = p.lower()
            if pl in ("tests", "test", "testing", "conftest"):
                return True
            if pl.startswith("test_") or pl.endswith("_test"):
                return True
        return False

    def _walk(path, prefix):
        try:
            for item in pkgutil.iter_modules(path, prefix):
                if _excluded(item.name):
                    continue
                if item.name.split(".")[-1].startswith("_") and item.name != f"_{package_name}":
                    continue
                try:
                    mod = importlib.import_module(item.name)
                    modules[item.name] = mod
                    sub_path = getattr(mod, "__path__", None)
                    if item.ispkg and sub_path:
                        _walk(sub_path, item.name + ".")
                except (Exception, BaseException):
                    continue
        except (Exception, BaseException):
            pass

    if pkg_path:
        _walk(pkg_path, package_name + ".")
    return modules


# =====================================================================
# 2. Adapter protocol + implementations
# =====================================================================

class LibraryAdapter(abc.ABC):
    """
    Contract: given an importable library, provide the evidence the unified
    harvest pipeline needs, according to how the library is implemented.
    Adapters NEVER decide schema semantics (stage/type/state classification);
    they only supply raw evidence channels they physically possess.
    """
    name: str = "base"

    def can_adapt(self, root_module: Any) -> bool:
        return True

    # -- evidence channels (default implementations: introspection generic) --

    def enumerate_modules(self, root_module: Any) -> Dict[str, Any]:
        return _walk_package(root_module)

    def get_source(self, obj: Any) -> Optional[str]:
        return None

    def get_docstring(self, obj: Any) -> Optional[str]:
        try:
            return inspect.getdoc(obj)
        except Exception:
            return None

    def resolve_callable_signature(self, obj: Any, callable_name: str = "",
                                   parent_cls_name: str = "", mod: Any = None):
        """Signature resolution is shared: every adapter routes through the
        layered signature_introspector (runtime → text_signature → stubs → docs)."""
        return resolve_signature(obj, callable_name=callable_name,
                                 parent_cls_name=parent_cls_name, mod=mod)

    def describe(self) -> Dict[str, Any]:
        return {"adapter": self.name}


class PurePythonSourceAdapter(LibraryAdapter):
    """
    For libraries distributed as Python source (the most common kind):
    every evidence channel is available, including function source text,
    which enables AST-level dataflow evidence (mutator detection,
    receiver-state dependency analysis) in the harvest pipeline.
    """
    name = "pure_python_source"

    def can_adapt(self, root_module: Any) -> bool:
        if not _is_python_source_module(root_module):
            # A package whose __init__ is thin but whose submodules are .py
            pkg_path = getattr(root_module, "__path__", None)
            if not pkg_path:
                return False
            for child in Path(list(pkg_path)[0]).glob("*.py"):
                return True
            return False
        return True

    def get_source(self, obj: Any) -> Optional[str]:
        try:
            src = inspect.getsource(obj)
            return src if src and src.strip() else None
        except Exception:
            return None


class TypingStubAdapter(LibraryAdapter):
    """
    For compiled extension libraries that ship PEP 484 typing stubs (.pyi):
    signatures and return annotations (incl. None/Self mutator evidence declared
    in stubs) come from the stub AST; docstrings come from the runtime module.
    No Python source exists, so source-level dataflow evidence is declined.
    """
    name = "typing_stub"

    def can_adapt(self, root_module: Any) -> bool:
        try:
            if _find_stub_file_for_module(root_module) is not None:
                return True
        except Exception:
            pass
        # Package-shipped stubs: any .pyi beside or inside the package
        pkg_path = getattr(root_module, "__path__", None) or []
        init_dir = Path(pkg_path[0]) if pkg_path else None
        if init_dir and init_dir.is_dir():
            try:
                if any(init_dir.glob("*.pyi")):
                    return True
            except Exception:
                pass
        return False

    def get_source(self, obj: Any) -> Optional[str]:
        # Compiled library: no Python source. Stub bodies are declarations
        # (`...`) and carry no runtime dataflow to analyze.
        return None

    def get_docstring(self, obj: Any) -> Optional[str]:
        doc = super().get_docstring(obj)
        if doc:
            return doc
        # Many compiled extensions embed a signature line in __doc__;
        # the docstring channel still serves it verbatim (tier-4 parsing
        # happens inside resolve_signature).
        raw_doc = getattr(obj, "__doc__", None)
        return raw_doc if isinstance(raw_doc, str) and raw_doc.strip() else None


class RuntimeReflectionAdapter(LibraryAdapter):
    """
    Universal fallback: any importable library. Evidence is limited to live
    object introspection (dir, docstrings, runtime signatures). The pipeline
    treats missing evidence honestly: unknown returns stay composable stage-2
    morphisms, no fabricated states.
    """
    name = "runtime_reflection"


class CompositeLibraryAdapter(LibraryAdapter):
    """
    Probes the library once and chains concrete adapters from richest to
    poorest evidence. Every evidence request cascades: the first adapter that
    can answer wins. Hybrid libraries therefore get source evidence for their
    Python layer and stub/docstring evidence for their compiled layer.
    """
    name = "composite"

    #: richness order — first probe wins for can_adapt; per-request cascade
    _ADAPTER_CLASSES = (PurePythonSourceAdapter, TypingStubAdapter, RuntimeReflectionAdapter)

    def __init__(self, root_module: Any):
        self.root_module = root_module
        self.chain: List[LibraryAdapter] = []
        for cls in self._ADAPTER_CLASSES:
            try:
                adapter = cls()
                if adapter.can_adapt(root_module):
                    self.chain.append(adapter)
            except Exception:
                continue
        if not self.chain:
            self.chain = [RuntimeReflectionAdapter()]

    # -- cascade channels --

    def enumerate_modules(self, root_module: Any = None) -> Dict[str, Any]:
        return _walk_package(root_module or self.root_module)

    def get_source(self, obj: Any) -> Optional[str]:
        for adapter in self.chain:
            src = adapter.get_source(obj)
            if src:
                return src
        return None

    def get_docstring(self, obj: Any) -> Optional[str]:
        for adapter in self.chain:
            doc = adapter.get_docstring(obj)
            if doc:
                return doc
        return None

    def resolve_callable_signature(self, obj: Any, callable_name: str = "",
                                   parent_cls_name: str = "", mod: Any = None):
        for adapter in self.chain:
            sig = adapter.resolve_callable_signature(obj, callable_name, parent_cls_name, mod)
            if sig is not None:
                return sig
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "adapter": self.name,
            "chain": [a.name for a in self.chain],
            "has_source_evidence": any(isinstance(a, PurePythonSourceAdapter) for a in self.chain),
            "has_stub_evidence": any(isinstance(a, TypingStubAdapter) for a in self.chain),
        }


_adapter_cache: Dict[str, CompositeLibraryAdapter] = {}


def get_adapter_for_package(package_name: str) -> CompositeLibraryAdapter:
    """
    Adapter selection by implementation kind. Probes the importable package
    once and returns the composite evidence provider. Raises ImportError when
    the package cannot be imported at all (the pipeline cannot run without
    the library installed).
    """
    key = package_name
    if key in _adapter_cache:
        return _adapter_cache[key]

    try:
        root_module = importlib.import_module(package_name)
    except ImportError:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        root_module = importlib.import_module(package_name)

    adapter = CompositeLibraryAdapter(root_module)
    logger.info(f"[ADAPTER] {package_name}: chain={adapter.describe()['chain']}")
    _adapter_cache[key] = adapter
    return adapter
