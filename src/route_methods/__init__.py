"""
src/route_methods/__init__.py - Neuro-Symbolic Topological Lattice (NSTL)
Pluggable RouteMethods registry and factory (M0 - M6).
"""

from typing import Dict, Type, Optional

from .base import RouteMethod
from .m0_trellis import M0TrellisRouteMethod
from .m1_clause_anchor import M1ClauseAnchorRouteMethod
from .m2_endpoint_anchor import M2EndpointAnchorRouteMethod
from .m3_greedy_freeze import M3GreedyFreezeRouteMethod
from .m4_llm_stepwise import M4LLMStepwiseRouteMethod
from .m5_llm_oneshot import M5LLMOneShotRouteMethod
from .m6_hybrid_anchors import M6HybridAnchorsRouteMethod

ROUTE_METHOD_REGISTRY: Dict[str, Type[RouteMethod]] = {
    "m0": M0TrellisRouteMethod,
    "m0_trellis": M0TrellisRouteMethod,
    "trellis": M0TrellisRouteMethod,
    "viterbi": M0TrellisRouteMethod,

    "m1": M1ClauseAnchorRouteMethod,
    "m1_clause_anchor": M1ClauseAnchorRouteMethod,
    "clause": M1ClauseAnchorRouteMethod,
    "clause_anchor": M1ClauseAnchorRouteMethod,

    "m2": M2EndpointAnchorRouteMethod,
    "m2_endpoint_anchor": M2EndpointAnchorRouteMethod,
    "endpoint": M2EndpointAnchorRouteMethod,
    "endpoint_anchor": M2EndpointAnchorRouteMethod,

    "m3": M3GreedyFreezeRouteMethod,
    "m3_greedy_freeze": M3GreedyFreezeRouteMethod,
    "greedy": M3GreedyFreezeRouteMethod,
    "greedy_freeze": M3GreedyFreezeRouteMethod,

    "m4": M4LLMStepwiseRouteMethod,
    "m4_llm_stepwise": M4LLMStepwiseRouteMethod,
    "llm_stepwise": M4LLMStepwiseRouteMethod,

    "m5": M5LLMOneShotRouteMethod,
    "m5_llm_oneshot": M5LLMOneShotRouteMethod,
    "llm_oneshot": M5LLMOneShotRouteMethod,

    "m6": M6HybridAnchorsRouteMethod,
    "m6_hybrid_anchors": M6HybridAnchorsRouteMethod,
    "hybrid": M6HybridAnchorsRouteMethod,
    "hybrid_anchors": M6HybridAnchorsRouteMethod,
}


def get_route_method(name: str = "m0", **kwargs) -> RouteMethod:
    """
    Factory function returning an instantiated RouteMethod by name or alias.
    """
    clean_name = str(name).strip().lower()
    cls = ROUTE_METHOD_REGISTRY.get(clean_name)
    if not cls:
        raise ValueError(
            f"Unknown route method '{name}'. Available: {list(ROUTE_METHOD_REGISTRY.keys())}"
        )
    return cls(**kwargs)


__all__ = [
    "RouteMethod",
    "M0TrellisRouteMethod",
    "M1ClauseAnchorRouteMethod",
    "M2EndpointAnchorRouteMethod",
    "M3GreedyFreezeRouteMethod",
    "M4LLMStepwiseRouteMethod",
    "M5LLMOneShotRouteMethod",
    "M6HybridAnchorsRouteMethod",
    "ROUTE_METHOD_REGISTRY",
    "get_route_method",
]
