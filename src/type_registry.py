"""Centralized type-to-domain registry built dynamically from loaded cells.

Replaces hardcoded type→domain string mappings (VIO-09, VIO-52) with a
data-driven registry that is built once at startup from the actual cell
population.
"""
from typing import Dict, Set, Optional, List, TYPE_CHECKING
from log_config import get_logger

if TYPE_CHECKING:
    from lattice import Cell

logger = get_logger('type_registry')

_WELL_KNOWN_ALIASES = {
    'pd': 'pandas', 'np': 'numpy', 'cv2': 'opencv', 'plt': 'matplotlib',
    'sns': 'seaborn', 'tf': 'tensorflow', 'sk': 'scikit', 'sklearn': 'scikit',
    'pytorch': 'torch',
}

class TypeRegistry:
    def __init__(self):
        self._type_to_domains: Dict[str, Set[str]] = {}
        self._domain_aliases: Dict[str, str] = _WELL_KNOWN_ALIASES.copy()

    @classmethod
    def build(cls, cells: List['Cell']) -> 'TypeRegistry':
        registry = cls()
        for cell in cells:
            domain = cell.domain_name
            if not domain:
                continue
                
            canonical_domain = registry.resolve_alias(domain)
            
            # Extract types from inputs and outputs
            types_used = []
            if hasattr(cell.inputs, 'type_name'):
                types_used.append(cell.inputs.type_name)
            if hasattr(cell.outputs, 'type_name'):
                types_used.append(cell.outputs.type_name)
                
            for t in types_used:
                if not t:
                    continue
                if t not in registry._type_to_domains:
                    registry._type_to_domains[t] = set()
                registry._type_to_domains[t].add(canonical_domain)
                
        logger.info(f"Built TypeRegistry with {len(registry._type_to_domains)} types mapped to domains.")
        return registry

    def resolve_alias(self, alias: str) -> str:
        """Returns the canonical domain name for an alias, or the alias itself if unknown."""
        if not alias:
            return alias
        return self._domain_aliases.get(alias.lower(), alias.lower())

    def domains_for_type(self, type_name: str) -> Set[str]:
        """
        Returns the set of domains associated with a type.
        Resolves aliases implicitly by returning canonical domains.
        Returns empty set for unknown types.
        """
        return self._type_to_domains.get(type_name, set())

    def is_domain_compatible(self, cell_domain: str, expected_type: str) -> float:
        """
        Returns a compatibility factor:
          1.0 if cell_domain is in domains_for_type(expected_type) or expected_type is 'any'
          0.5 if no domain information is available (neutral)
          0.01 if cell_domain conflicts with expected_type's domains
        """
        if expected_type == 'any':
            return 1.0
            
        canonical_cell_domain = self.resolve_alias(cell_domain or '')

        # Domain-specific typestate invariants
        if expected_type in ('DataFrame', 'Series', 'pd.DataFrame', 'DataFrame_Object'):
            if canonical_cell_domain in ('opencv', 'cv2', 'image_processing'):
                return 0.01
            if canonical_cell_domain in ('pandas', 'scikit', 'scipy', 'numpy', 'python', 'core', 'data_engineering', 'ml', 'generic', 'synthesized_domain'):
                return 1.0

        if expected_type in ('Mat', 'Image', 'cv2.Mat', 'Mat_Object'):
            if canonical_cell_domain in ('pandas', 'data_engineering'):
                return 0.01
            if canonical_cell_domain in ('opencv', 'cv2', 'numpy', 'python', 'core', 'image_processing', 'image', 'generic', 'synthesized_domain'):
                return 1.0

        expected_domains = self.domains_for_type(expected_type)
        if not expected_domains:
            return 0.5
            
        if not canonical_cell_domain:
            return 0.5
            
        if canonical_cell_domain in expected_domains:
            return 1.0
            
        return 0.01
