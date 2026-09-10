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
    from universal_harvester import UniversalHarvester
except ImportError:
    from .schema import CellSchema, PortSchema, TreeSchema
    from .universal_harvester import UniversalHarvester


class IntelligentHarvester(UniversalHarvester):
    """
    Universal library harvester delegating purely to category-theoretic UniversalHarvester reflection.
    Fully domain-agnostic, zero hardcoded word lists.
    """

    def __init__(self, domain: str, package_name: Optional[str] = None):
        super().__init__(domain_name=domain, package_name=package_name)
        self.domain = domain

