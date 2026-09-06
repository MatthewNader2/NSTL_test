# src/schema.py
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import ast
import re

class PortSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type_name: str
    state: str = "default"
    qualifiers: List[List[str]] = Field(default_factory=list)
    default_value: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    required: bool = True

class CellSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cell_id: str
    stage: Literal[0, 1, 2, 3]
    inputs: Dict[str, PortSchema] = Field(default_factory=dict)
    outputs: Dict[str, PortSchema] = Field(default_factory=dict)
    slots: Dict[str, Any] = Field(default_factory=dict)
    code_template: str
    dependencies: List[str] = Field(default_factory=list)
    semantic_tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    docstring: Optional[str] = ""
    enrichment_source: Optional[str] = None   # "docs" | "llm" | None (curated/native)
    enriched_at: Optional[str] = None         # ISO8601 timestamp, set when enrichment_source is set
    domain_name: Optional[str] = None
    node_type: Optional[str] = "function"
    node_role: Optional[str] = "function"
    verified: bool = True
    source_priority: int = 100  # 1 = curated seed, 100 = auto-harvested

    @property
    def primary_input(self) -> PortSchema:
        if not self.inputs:
            return PortSchema(type_name="any", state="any")
        try:
            from .lattice import TypeRegistry
        except (ImportError, ValueError):
            from lattice import TypeRegistry
        registry = TypeRegistry.get_instance()
        if self.stage == 1:
            # Ingestion takes primitive / environment input
            for p in self.inputs.values():
                if not registry.is_container_type(p.type_name):
                    return p
            return next(iter(self.inputs.values()))
        # Stage 2 and 3: prioritize carrier / container types
        for p in self.inputs.values():
            if registry.is_container_type(p.type_name):
                return p
        return next(iter(self.inputs.values()))

    @property
    def primary_output(self) -> PortSchema:
        if not self.outputs:
            return PortSchema(type_name="None", state="default")
        try:
            from .lattice import TypeRegistry
        except (ImportError, ValueError):
            from lattice import TypeRegistry
        registry = TypeRegistry.get_instance()
        for p in self.outputs.values():
            if registry.is_container_type(p.type_name):
                return p
        return next(iter(self.outputs.values()))

    @field_validator("code_template")
    @classmethod
    def validate_template_syntax(cls, v: str) -> str:
        # Dry-run AST parse with dummy variables to ensure syntactically valid Python
        # Use unique placeholder names to avoid false positive validation
        seen = {}
        def _replace_ph(m):
            name = m.group(0)
            if name not in seen:
                seen[name] = f"_ph_{len(seen)}"
            return seen[name]
        dummy_code = re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", _replace_ph, v)
        try:
            ast.parse(dummy_code)
        except SyntaxError as e:
            raise ValueError(f"Invalid code_template syntax: {v}. Error: {e}")
        return v

class TreeSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    domain: str
    version: str = "1.0.0"
    cells: List[CellSchema]
