# src/schema.py
from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import ast
import re

class ConditionPredicate(BaseModel):
    """
    Formal predicate for preconditions and postconditions/effects.
    Domain-agnostic: supports arbitrary properties, operators, and target ports/variables.
    Examples: channels==1, dtype=='binary', is_fitted==True, shape[-1]==3.
    """
    model_config = ConfigDict(extra="ignore")

    target: Optional[str] = None       # Port name or variable identifier (e.g. "image", "df", "self")
    property: Optional[str] = None     # Property being evaluated (e.g. "channels", "dtype", "is_fitted")
    operator: str = "=="               # "==", "!=", ">", ">=", "<", "<=", "in", "is", "matches"
    value: Any = None                  # Target value (e.g. 1, "binary", True)
    expression: Optional[str] = None   # Raw string expression (e.g. "channels == 1")
    description: Optional[str] = None  # Human-readable explanation

    @classmethod
    def from_any(cls, v: Any) -> "ConditionPredicate":
        if isinstance(v, ConditionPredicate):
            return v
        if isinstance(v, str):
            expr = v.strip()
            match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_\.]*)\s*(==|!=|>=|<=|>|<|in|is)\s*(.+)$", expr)
            if match:
                prop = match.group(1)
                op = match.group(2)
                val_raw = match.group(3).strip()
                val: Any = val_raw
                if val_raw.lower() == "true":
                    val = True
                elif val_raw.lower() == "false":
                    val = False
                elif (val_raw.startswith("'") and val_raw.endswith("'")) or (val_raw.startswith('"') and val_raw.endswith('"')):
                    val = val_raw[1:-1]
                else:
                    try:
                        val = int(val_raw)
                    except ValueError:
                        try:
                            val = float(val_raw)
                        except ValueError:
                            pass
                return cls(property=prop, operator=op, value=val, expression=expr)
            return cls(expression=expr)
        if isinstance(v, dict):
            if "property" in v or "operator" in v or "expression" in v or "target" in v:
                return cls(**v)
            items = list(v.items())
            if len(items) == 1:
                return cls(property=items[0][0], operator="==", value=items[0][1])
            return cls(property=items[0][0], operator="==", value=items[0][1], description=str(v))
        return cls(description=str(v))

class EdgeSchema(BaseModel):
    """
    Explicit transition edge between lattice cells.
    Captures learned/curated transition affinities and bridging contracts.
    """
    model_config = ConfigDict(extra="ignore")

    target_cell_id: str
    affinity_score: float = Field(default=1.0, ge=0.0)
    bridging_precondition: Optional[Union[ConditionPredicate, str, Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("bridging_precondition", mode="before")
    @classmethod
    def normalize_bridging_precondition(cls, v: Any) -> Optional[Any]:
        if v is None:
            return None
        if isinstance(v, (str, dict)):
            return ConditionPredicate.from_any(v)
        return v

class TypestateDefinition(BaseModel):
    """Declarative typestate specification within a domain."""
    model_config = ConfigDict(extra="ignore")

    name: str                                  # e.g. "color", "gray", "raw", "unfit"
    parent_state: Optional[str] = None         # For state hierarchy / subtyping
    carrier_type: Optional[str] = None         # e.g. "ndarray", "DataFrame", "BaseEstimator"
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict) # e.g. {"channels": 1}

class TypestateVocabularySchema(BaseModel):
    """Domain-level typestate vocabulary and state transitions."""
    model_config = ConfigDict(extra="ignore")

    domain: str
    states: List[Union[str, TypestateDefinition]] = Field(default_factory=list)
    transitions: List[Dict[str, Any]] = Field(default_factory=list) # optional e.g. [{"from": "color", "to": "gray", "via": "cvtColor"}]
    initial_state: Optional[str] = None
    terminal_states: List[str] = Field(default_factory=list)

class PortSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type_name: str
    state: str = "default"
    parent_state: Optional[str] = None
    accepted_states: List[str] = Field(default_factory=list)
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

    # --- Semantic IR Extensions (Phase 1) ---
    preconditions: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)
    postconditions: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)
    effects: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)
    edges: List[Union[EdgeSchema, Dict[str, Any]]] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        # Synchronize effects and postconditions if one is set but not the other
        if self.postconditions and not self.effects:
            self.effects = list(self.postconditions)
        elif self.effects and not self.postconditions:
            self.postconditions = list(self.effects)

    @field_validator("preconditions", mode="before")
    @classmethod
    def normalize_preconditions(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, dict):
            if "property" in v or "expression" in v or "target" in v:
                return [ConditionPredicate.from_any(v)]
            return [ConditionPredicate(property=k, operator="==", value=val) for k, val in v.items()]
        if isinstance(v, (list, tuple)):
            res = []
            for item in v:
                if isinstance(item, (str, dict, ConditionPredicate)):
                    res.append(ConditionPredicate.from_any(item) if not isinstance(item, ConditionPredicate) else item)
                else:
                    res.append(item)
            return res
        return [ConditionPredicate.from_any(v)]

    @field_validator("postconditions", "effects", mode="before")
    @classmethod
    def normalize_conditions_list(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, dict):
            if "property" in v or "expression" in v or "target" in v:
                return [ConditionPredicate.from_any(v)]
            return [ConditionPredicate(property=k, operator="==", value=val) for k, val in v.items()]
        if isinstance(v, (list, tuple)):
            res = []
            for item in v:
                if isinstance(item, (str, dict, ConditionPredicate)):
                    res.append(ConditionPredicate.from_any(item) if not isinstance(item, ConditionPredicate) else item)
                else:
                    res.append(item)
            return res
        return [ConditionPredicate.from_any(v)]

    @field_validator("edges", mode="before")
    @classmethod
    def normalize_edges(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, dict):
            if "target_cell_id" in v:
                return [EdgeSchema(**v)]
            res = []
            for target_id, edge_info in v.items():
                if isinstance(edge_info, dict):
                    res.append(EdgeSchema(target_cell_id=target_id, **edge_info))
                elif isinstance(edge_info, (int, float)):
                    res.append(EdgeSchema(target_cell_id=target_id, affinity_score=float(edge_info)))
                else:
                    res.append(EdgeSchema(target_cell_id=target_id))
            return res
        if isinstance(v, (list, tuple)):
            res = []
            for item in v:
                if isinstance(item, dict):
                    res.append(EdgeSchema(**item))
                elif isinstance(item, EdgeSchema):
                    res.append(item)
                else:
                    res.append(item)
            return res
        return []

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
    typestates: Optional[Union[TypestateVocabularySchema, Dict[str, Any], List[str]]] = None

    @field_validator("typestates", mode="before")
    @classmethod
    def normalize_typestates(cls, v: Any) -> Optional[Any]:
        if v is None:
            return None
        if isinstance(v, TypestateVocabularySchema):
            return v
        if isinstance(v, dict):
            if "domain" in v:
                return TypestateVocabularySchema(**v)
            return v
        if isinstance(v, (list, tuple)):
            return TypestateVocabularySchema(domain="generic", states=list(v))
        return v
