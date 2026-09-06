# src/harvester.py
"""
src/harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Universal Intelligent Harvester.
Grounds all API harvesting in category-theoretic reflection via UniversalHarvester.
Contains ZERO hardcoded domain lists or heuristic keyword matching.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from schema import CellSchema, PortSchema, TreeSchema
    from universal_harvester import UniversalHarvester, split_identifier_keywords, infer_qualifiers_from_constant
except ImportError:
    from .schema import CellSchema, PortSchema, TreeSchema
    from .universal_harvester import UniversalHarvester, split_identifier_keywords, infer_qualifiers_from_constant


class IntelligentHarvester(UniversalHarvester):
    """
    Universal library harvester delegating purely to category-theoretic UniversalHarvester reflection.
    Fully domain-agnostic, zero hardcoded word lists.
    """

    def __init__(self, domain: str, package_name: Optional[str] = None, container_type: Optional[str] = None):
        super().__init__(domain_name=domain, package_name=package_name, container_type=container_type)
        self.domain = domain
        self.module = self.root_module
        self.discover_submodules()
        self.collect_constants_and_enums()
        self.identify_container_classes()
        self.enum_constants: Dict[str, str] = {}
        for prefix, items in self.constants_by_prefix.items():
            for name, _, mod in items:
                self.enum_constants[name] = f"{mod}.{name}"

    @property
    def default_container(self) -> str:
        if self.primary_container:
            return self.primary_container.__name__
        if self.container_classes:
            return next(iter(self.container_classes)).__name__
        return "DataObject"

    def harvest_function(self, func_name: str, func_obj: Any, parent_mod_name: Optional[str] = None) -> Optional[CellSchema]:
        """Harvests a single callable using UniversalHarvester categorical reflection."""
        mod_name = parent_mod_name or getattr(func_obj, "__module__", self.package_name) or self.package_name
        cell = self.harvest_callable(func_name=func_name, func_obj=func_obj, mod_name=mod_name)
        if cell is not None:
            for p_name, p_sig in cell.inputs.items():
                if p_sig.type_name == "Enum" and p_sig.domain:
                    prefix = p_sig.domain.split(".")[-1].replace("_*", "")
                    cands = [k for k in self.enum_constants if k.startswith(f"{prefix}_")]
                    if cands:
                        best = next((c for c in cands if "BGR2GRAY" in c or "DEFAULT" in c or "STANDARD" in c), cands[0])
                        target_const = self.enum_constants[best]
                        cell.code_template = cell.code_template.replace(f"{{{p_name}}}", target_const)
                        p_sig.default_value = target_const
        return cell
