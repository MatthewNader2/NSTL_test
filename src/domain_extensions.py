"""
src/domain_extensions.py - Domain Extension Registry for NSTL Unification Engine.

Provides a thread-safe singleton registry for domain-specific extensions:
- Column access renderers (how to access a column from a table carrier)
- Semantic flag patterns (prompt keyword → operational flag mappings)
- Enum resolvers (domain-qualified enum constant resolution)
- Verification contract plugins (domain-specific postcondition augmentation)

The core unification engine stays domain-agnostic; all domain vocabulary
and library-specific knowledge is registered here by plugin modules.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Protocol, Set, Tuple,
    runtime_checkable,
)


# =====================================================================
# Column Access Renderers
# =====================================================================

class ColumnAccessRenderer(ABC):
    """Strategy for rendering column access expressions on tabular carriers."""

    @abstractmethod
    def render(self, var: str, col: str) -> str:
        """Return a code expression that accesses column `col` on variable `var`."""
        ...


class SubscriptRenderer(ColumnAccessRenderer):
    """Default renderer: ``var['col']``."""

    def render(self, var: str, col: str) -> str:
        return f"{var}['{col}']"


class PandasRenderer(ColumnAccessRenderer):
    """Pandas-specific renderer: ``var['col'].to_numpy()`` for tensor ports."""

    def render(self, var: str, col: str) -> str:
        return f"{var}['{col}'].to_numpy()"


# =====================================================================
# Semantic Flag Patterns
# =====================================================================

@dataclass(frozen=True)
class SemanticFlagPattern:
    """
    Maps a set of prompt keywords to an operational flag.

    keywords:  frozenset of lowercase trigger words
    flag_key:  name of the flag to set (e.g. "ascending", "is_hsv")
    flag_val:  value to assign when any keyword matches
    extras:    additional flags to set alongside the primary flag
    """
    keywords: FrozenSet[str]
    flag_key: str
    flag_val: Any
    extras: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Enum Resolvers
# =====================================================================

@dataclass
class EnumResolverEntry:
    """
    Registry entry for domain-specific enum constant resolution.

    match_fn:   (placeholder_name: str) -> bool
                Returns True if this resolver handles the given placeholder.
    resolve_fn: (placeholder: str, template: str, flags: Dict) -> Optional[str]
                Returns the resolved constant string, or None if unresolvable.
    """
    match_fn: Callable[[str], bool]
    resolve_fn: Callable[[str, str, Dict[str, Any]], Optional[str]]


# =====================================================================
# Verification Contract Plugins
# =====================================================================

@runtime_checkable
class VerificationContractPlugin(Protocol):
    """
    Protocol for domain-specific verification contract augmentation.

    Plugins inspect pipeline bindings and context to produce additional
    cell_checks and terminal_checks that get merged into the final
    VerificationContract.
    """

    def augment_contract(
        self,
        pipeline_bindings: List[Tuple[Any, Dict[str, str]]],
        ctx: Any,
        cell_checks: List[Dict[str, Any]],
        terminal_checks: List[Dict[str, Any]],
    ) -> None:
        """
        Append domain-specific checks to the mutable ``cell_checks`` and
        ``terminal_checks`` lists.  Called during contract construction
        before the lists are frozen into the immutable VerificationContract.
        """
        ...


# =====================================================================
# Domain Extension Registry (Thread-Safe Singleton)
# =====================================================================

class DomainExtensionRegistry:
    """
    Central registry for all domain-specific extensions.

    Thread-safe: all mutations are guarded by an internal lock.
    The singleton instance is available as the module-level ``EXTENSIONS``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._column_renderers: Dict[str, ColumnAccessRenderer] = {}
        self._default_renderer: ColumnAccessRenderer = SubscriptRenderer()
        self._semantic_flags: List[SemanticFlagPattern] = []
        self._enum_resolvers: List[EnumResolverEntry] = []
        self._verification_plugins: List[VerificationContractPlugin] = []

    # -- Column Renderers --

    def register_column_renderer(self, port_type_key: str, renderer: ColumnAccessRenderer) -> None:
        """Register a renderer for a specific port type (e.g. 'tensor')."""
        with self._lock:
            self._column_renderers[port_type_key.lower()] = renderer

    def resolve_column_renderer(self, port_type: str) -> ColumnAccessRenderer:
        """Return the best renderer for the given port type, or the default."""
        with self._lock:
            # Check exact match first
            renderer = self._column_renderers.get(port_type.lower())
            if renderer is not None:
                return renderer
            # Check prefix/subtype matches
            for key, r in self._column_renderers.items():
                if key in port_type.lower():
                    return r
            return self._default_renderer

    # -- Semantic Flags --

    def register_semantic_flag(self, pattern: SemanticFlagPattern) -> None:
        """Register a semantic flag pattern."""
        with self._lock:
            self._semantic_flags.append(pattern)

    def get_semantic_flags(self) -> List[SemanticFlagPattern]:
        """Return all registered semantic flag patterns (snapshot)."""
        with self._lock:
            return list(self._semantic_flags)

    # -- Enum Resolvers --

    def register_enum_resolver(self, entry: EnumResolverEntry) -> None:
        """Register a domain-specific enum resolver."""
        with self._lock:
            self._enum_resolvers.append(entry)

    def resolve_enum(self, placeholder: str, template: str, flags: Dict[str, Any]) -> Optional[str]:
        """Try all registered enum resolvers for the given placeholder."""
        with self._lock:
            resolvers = list(self._enum_resolvers)
        for entry in resolvers:
            if entry.match_fn(placeholder.lower()):
                result = entry.resolve_fn(placeholder, template, flags)
                if result is not None:
                    return result
        return None

    # -- Verification Plugins --

    def register_verification_plugin(self, plugin: VerificationContractPlugin) -> None:
        """Register a verification contract augmentation plugin."""
        with self._lock:
            self._verification_plugins.append(plugin)

    def get_verification_plugins(self) -> List[VerificationContractPlugin]:
        """Return all registered verification plugins (snapshot)."""
        with self._lock:
            return list(self._verification_plugins)


# Module-level singleton
EXTENSIONS = DomainExtensionRegistry()
