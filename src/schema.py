# src/schema.py
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import ast

class PortSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type_name: str
    state: str = "default"
    qualifiers: List[List[str]] = Field(default_factory=list)
    default_value: Optional[Any] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    required: bool = True
    abstract_type: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    param_kind: Optional[str] = "standard"  # "positional_only", "keyword_only", "var_positional", "var_keyword", "standard"
    value_constraints: Optional[Dict[str, Any]] = None  # e.g. {"min": 0, "max": 1, "interval": "[0, 1]"}
    shape_contract: Optional[Dict[str, Any]] = None  # e.g. {"ndim": 2}

    @field_validator("abstract_type", mode="before")
    @classmethod
    def clean_abstract_type(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        v_str = str(v).strip()
        if v_str.lower() in ("", "none", "null"):
            return None
        return v_str

class CellSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cell_id: str
    stage: Literal[0, 1, 2, 3]
    inputs: Dict[str, PortSchema] = Field(default_factory=dict)
    outputs: Dict[str, PortSchema] = Field(default_factory=dict)
    topology_type: str = "sequential"  # "sequential", "monoidal_product", "coproduct_branch", "traced_loop"
    slots: Dict[str, Any] = Field(default_factory=dict)
    feedback_state_type: Optional[str] = None
    bound_slots: Dict[str, Any] = Field(default_factory=dict)
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
    is_public: bool = True
    mutation_type: str = "pure"  # "pure" | "in_place"
    is_context_manager: bool = False
    raises: List[str] = Field(default_factory=list)
    type_vars: List[str] = Field(default_factory=list)

    @field_validator("topology_type")
    @classmethod
    def validate_topology(cls, v: str) -> str:
        allowed = {"sequential", "monoidal_product", "coproduct_branch", "traced_loop"}
        if v not in allowed:
            raise ValueError(f"Invalid topology_type: '{v}'. Must be one of {allowed}")
        return v

    @property
    def primary_input(self) -> PortSchema:
        if not self.inputs:
            return PortSchema(type_name="any", state="any")
        required = [p for p in self.inputs.values() if p.required]
        if required:
            non_scalar = [p for p in required if (p.abstract_type or "").lower() not in ("scalar", "text", "path", "logical")]
            return non_scalar[0] if non_scalar else required[0]
        return next(iter(self.inputs.values()))

    @property
    def primary_output(self) -> PortSchema:
        if not self.outputs:
            return PortSchema(type_name="any", state="default")
        return next(iter(self.outputs.values()))

    @field_validator("code_template")
    @classmethod
    def validate_template_syntax(cls, v: str) -> str:
        # Dry-run AST parse with dummy variables to ensure syntactically valid Python
        # Use unique placeholder names to avoid false positive validation
        seen: Dict[str, str] = {}
        res = []
        i = 0
        n = len(v)
        while i < n:
            if v[i] == "{" and i + 1 < n:
                end = v.find("}", i + 1)
                if end != -1:
                    inner = v[i + 1 : end]
                    if inner.isidentifier():
                        key = f"{{{inner}}}"
                        if key not in seen:
                            seen[key] = f"_ph_{len(seen)}"
                        res.append(seen[key])
                        i = end + 1
                        continue
            res.append(v[i])
            i += 1
        dummy_code = "".join(res)
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
