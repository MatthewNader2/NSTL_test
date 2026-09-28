# src/schema.py
from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import ast


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
            if not expr:
                return cls(expression=expr)

            # Robust AST-based parsing.  Compound expressions are preserved as
            # `expression` instead of being silently truncated at the first operator.
            try:
                tree = ast.parse(expr, mode="eval")
            except SyntaxError:
                return cls(expression=expr)

            body = tree.body
            if isinstance(body, ast.Compare) and len(body.ops) == 1 and len(body.comparators) == 1:
                op_map = {
                    ast.Eq: "==",
                    ast.NotEq: "!=",
                    ast.Gt: ">",
                    ast.GtE: ">=",
                    ast.Lt: "<",
                    ast.LtE: "<=",
                    ast.In: "in",
                    ast.NotIn: "not in",
                    ast.Is: "is",
                    ast.IsNot: "is not",
                }
                op = op_map.get(type(body.ops[0]))
                if op:
                    left = body.left
                    try:
                        prop = ast.unparse(left)
                    except Exception:
                        prop = None
                    if prop and all(part.isidentifier() for part in prop.split(".")):
                        right = body.comparators[0]
                        try:
                            val = ast.literal_eval(right)
                        except Exception:
                            try:
                                val = ast.unparse(right)
                            except Exception:
                                val = None
                        return cls(
                            property=prop,
                            operator=op,
                            value=val,
                            expression=expr,
                        )

            # Compound / complex expressions are retained intact.
            return cls(expression=expr)

        if isinstance(v, dict):
            if "property" in v or "operator" in v or "expression" in v or "target" in v:
                return cls(**v)
            items = list(v.items())
            if len(items) == 1:
                return cls(property=items[0][0], operator="==", value=items[0][1])
            return cls(
                property=items[0][0],
                operator="==",
                value=items[0][1],
                description=str(v),
            )

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

    name: str
    parent_state: Optional[str] = None
    carrier_type: Optional[str] = None
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class TypestateVocabularySchema(BaseModel):
    """Domain-level typestate vocabulary and state transitions."""
    model_config = ConfigDict(extra="ignore")

    domain: str
    states: List[Union[str, TypestateDefinition]] = Field(default_factory=list)
    transitions: List[Dict[str, Any]] = Field(default_factory=list)
    initial_state: Optional[str] = None
    terminal_states: List[str] = Field(default_factory=list)


class PortSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type_name: str
    state: str = "default"
    parent_state: Optional[str] = None
    accepted_states: List[str] = Field(default_factory=list)
    qualifiers: List[Any] = Field(default_factory=list)
    default_value: Optional[Any] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    required: bool = True
    abstract_type: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    param_kind: Optional[str] = "standard"
    value_constraints: Optional[Dict[str, Any]] = None
    shape_contract: Optional[Union[Dict[str, Any], str]] = None
    port_role: Optional[str] = None
    role: Optional[str] = None
    polarity: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.role and not self.port_role:
            self.port_role = self.role
        elif self.port_role and not self.role:
            self.role = self.port_role

    @field_validator("accepted_states", mode="before")
    @classmethod
    def clean_accepted_states(cls, v: Any) -> List[str]:
        if v is None:
            return []
        return v

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

    # Stage 0 was never defined downstream.  Only 1, 2, 3 are valid.
    stage: Literal[1, 2, 3] = 2

    inputs: Dict[str, PortSchema] = Field(default_factory=dict)
    outputs: Dict[str, PortSchema] = Field(default_factory=dict)
    topology_type: str = "sequential"
    slots: Union[Dict[str, Any], List[str]] = Field(default_factory=list)
    feedback_state_type: Optional[str] = None
    bound_slots: Dict[str, Any] = Field(default_factory=dict)
    code_template: str = ""
    code_templates: Dict[str, str] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)

    # Canonical semantic fields.
    semantic_tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)  # alias for semantic_tags
    docstring: Optional[str] = ""
    doc: Optional[str] = None  # alias for docstring

    enrichment_source: Optional[str] = None
    enriched_at: Optional[str] = None
    domain_name: Optional[str] = None
    node_type: Optional[str] = "function"
    node_role: Optional[str] = "function"
    role: Optional[str] = None  # alias for node_role
    verified: bool = True
    source_priority: int = 100
    is_public: bool = True
    mutation_type: str = "pure"
    is_context_manager: bool = False
    raises: List[str] = Field(default_factory=list)
    type_vars: List[str] = Field(default_factory=list)
    endable: Optional[bool] = None
    primary_in: Optional[str] = None
    primary_out: Optional[str] = None

    # --- Macro Node Pipeline ---
    sub_cells: List[str] = Field(default_factory=list)
    internal_topology: Dict[str, List[str]] = Field(default_factory=dict)
    algorithmic_steps: List[str] = Field(default_factory=list)

    # --- Semantic IR Extensions ---
    preconditions: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)
    postconditions: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)
    effects: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)  # alias for postconditions
    edges: List[Union[EdgeSchema, Dict[str, Any]]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases_and_stage(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # doc -> docstring
        if "doc" in d:
            if not d.get("docstring"):
                d["docstring"] = d.pop("doc")
            else:
                d.pop("doc", None)

        # keywords -> semantic_tags
        if "keywords" in d:
            kw = d.pop("keywords") or []
            st = d.get("semantic_tags") or []
            if not st:
                d["semantic_tags"] = kw
            else:
                d["semantic_tags"] = list(dict.fromkeys(list(st) + list(kw)))

        # role -> node_role
        if "role" in d:
            if not d.get("node_role"):
                d["node_role"] = d.pop("role")
            else:
                d.pop("role", None)

        # effects -> postconditions
        if "effects" in d:
            ef = d.pop("effects") or []
            pc = d.get("postconditions") or []
            if not pc:
                d["postconditions"] = ef
            elif isinstance(pc, list) and isinstance(ef, list):
                d["postconditions"] = list(pc) + list(ef)

        # Stage 0 is invalid.
        if "stage" in d:
            v = d["stage"]
            if v is None:
                d["stage"] = 2
            elif isinstance(v, str) and v.isdigit():
                d["stage"] = int(v)
            if d.get("stage") == 0:
                raise ValueError("Stage 0 is undefined; valid stages are 1, 2, 3")
            if d.get("stage") not in (1, 2, 3):
                raise ValueError("Stage must be one of 1, 2, 3")

        return d

    @field_validator("inputs", mode="before")
    @classmethod
    def clean_inputs(cls, v: Any) -> Any:
        if isinstance(v, dict):
            cleaned = {}
            for k, val in v.items():
                if k == "dependencies" and isinstance(val, (list, tuple)):
                    continue
                cleaned[k] = val
            return cleaned
        return v

    def model_post_init(self, __context: Any) -> None:
        if not self.code_template and self.code_templates:
            self.code_template = self.code_templates.get(
                "python", next(iter(self.code_templates.values()), "")
            )
        if self.doc and not self.docstring:
            self.docstring = self.doc
        if self.docstring and not self.doc:
            self.doc = self.docstring
        if self.role and not self.node_role:
            self.node_role = self.role
        if self.node_role and not self.role:
            self.role = self.node_role
        if self.sub_cells and (not self.node_type or self.node_type == "function"):
            self.node_type = "macro"
            self.node_role = "macro"
            self.role = "macro"
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
                    res.append(
                        ConditionPredicate.from_any(item)
                        if not isinstance(item, ConditionPredicate)
                        else item
                    )
                else:
                    res.append(item)
            return res
        return [ConditionPredicate.from_any(v)]

    @field_validator("postconditions", mode="before")
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
                    res.append(
                        ConditionPredicate.from_any(item)
                        if not isinstance(item, ConditionPredicate)
                        else item
                    )
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
        if not v or not str(v).strip():
            return "sequential"
        return str(v).strip().lower()

    @property
    def primary_input(self) -> Optional[PortSchema]:
        if not self.inputs:
            return None
        if self.primary_in and self.primary_in in self.inputs:
            return self.inputs[self.primary_in]
        required = [p for p in self.inputs.values() if p.required]
        if required:
            non_scalar = [
                p
                for p in required
                if (p.abstract_type or "").lower() not in ("scalar", "text", "path", "logical")
            ]
            return non_scalar[0] if non_scalar else required[0]
        return next(iter(self.inputs.values()))

    @property
    def primary_output(self) -> Optional[PortSchema]:
        if not self.outputs:
            return None
        if self.primary_out and self.primary_out in self.outputs:
            return self.outputs[self.primary_out]
        return next(iter(self.outputs.values()))

    @field_validator("code_template", mode="before")
    @classmethod
    def validate_template_syntax(cls, v: Any) -> str:
        if v is None:
            return ""
        v_str = str(v)
        if not v_str.strip():
            return v_str

        seen: Dict[str, str] = {}
        res = []
        i = 0
        n = len(v_str)
        while i < n:
            if v_str[i] == "{" and i + 1 < n:
                end = v_str.find("}", i + 1)
                if end != -1:
                    inner = v_str[i + 1 : end]
                    if inner.isidentifier():
                        key = f"{{{inner}}}"
                        if key not in seen:
                            seen[key] = f"_ph_{len(seen)}"
                        res.append(seen[key])
                        i = end + 1
                        continue
            res.append(v_str[i])
            i += 1

        dummy_code = "".join(res)
        try:
            ast.parse(dummy_code)
        except SyntaxError as e:
            raise ValueError(f"Invalid code_template syntax: {v_str}. Error: {e}")
        return v_str


class TreeSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    domain: str
    version: str = "1.0.0"
    description: Optional[str] = None
    cells: List[CellSchema]
    types: Optional[Union[Dict[str, Any], List[Any]]] = Field(default_factory=dict)
    typestates: Optional[Union[TypestateVocabularySchema, Dict[str, Any], List[str]]] = None
    type_vars: List[str] = Field(default_factory=list)
    aliases: Dict[str, str] = Field(default_factory=dict)
    top_types: List[str] = Field(default_factory=list)
    top: Optional[Union[bool, List[str]]] = None
    product_constructors: List[str] = Field(default_factory=list)
    egress_intent_tokens: List[str] = Field(default_factory=list)
    polarity_hints: Dict[str, List[str]] = Field(default_factory=dict)
    artifact_readers: Dict[str, List[str]] = Field(default_factory=dict)
    advisory_qualifiers: List[List[str]] = Field(default_factory=list)
    abstract_carriers: List[str] = Field(default_factory=list)
    abstract_carrier_mapping: Dict[str, str] = Field(default_factory=dict)
    dest_port_tokens: List[str] = Field(default_factory=list)
    data_bearing_roles: List[str] = Field(default_factory=list)
    estimator_verbs: List[str] = Field(default_factory=list)
    column_projection_tokens: List[str] = Field(default_factory=list)
    asset_placeholders: Dict[str, str] = Field(default_factory=dict)
    output_placeholders: Dict[str, str] = Field(default_factory=dict)
    default_asset_placeholders: Dict[str, str] = Field(default_factory=dict)
    default_output_placeholders: Dict[str, str] = Field(default_factory=dict)
    stopwords: List[str] = Field(default_factory=list)
    preposition_triggers: List[str] = Field(default_factory=list)

    @field_validator("types", mode="before")
    @classmethod
    def normalize_types(cls, v: Any) -> Optional[Union[Dict[str, Any], List[Any]]]:
        if v is None:
            return {}
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return v
        return {}

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
