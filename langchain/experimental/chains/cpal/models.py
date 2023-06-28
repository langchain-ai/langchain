from __future__ import annotations  # allows pydantic model to reference itself

from typing import Optional, Any  # , Union
import networkx as nx
import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    validator,
    root_validator,
    PrivateAttr,
)  # , ValidationError
import duckdb
from langchain.experimental.chains.cpal.constants import Constant

# import pydantic

# from py_console import console
import re


class NarrativeModel(BaseModel):
    """One question can have many initial conditions."""

    story_outcome_question: str
    story_hypothetical: str
    story_plot: str

    @validator("*", pre=True)
    def empty_str_to_none(cls, v):
        """Empty strings are not allowed"""
        if v == "":
            return None
        return v


class EntityModel(BaseModel):
    """univariate math"""

    name: str = Field(description="entity name")
    code: str = Field(description="entity actions")
    value: float = Field(description="entity initial value")
    depends_on: list[str] = Field(default=[], description="ancestor entities")

    # TODO: depends_on_feedback_from (acyclic graph)
    # TODO: allow for multiple variables (multivariate math)

    class Config:
        validate_assignment = True

    @validator("name")
    def lower_case_name(cls, v):
        # is this a good idea?
        v = v.lower()
        return v


class CausalModel(BaseModel):
    attribute: str = Field(description="name of the attribute to be calculated")
    entities: list[EntityModel] = Field(description="entities in the story")

    # TODO: root validate each `entity.depends_on` using system's entity names


class EntitySettingModel(BaseModel):
    """
    Initial conditions for an entity

    {"name": "bud", "attribute": "pet_count", "value": 12}
    """

    name: str = Field(description="name of the entity")
    attribute: str = Field(description="name of the attribute to be calculated")
    value: float = Field(description="entity's attribute value (calculated)")

    @validator("name")
    def lower_case_transform(cls, v):
        v = v.lower()
        return v


class SystemSettingModel(BaseModel):
    """
    Initial global conditions for the system.

    {"parameter": "interest_rate", "value": .05}
    """

    parameter: str
    value: float


class InterventionModel(BaseModel):
    """
    aka initial conditions

    >>> intervention.dict()
    {
        entity_settings: [
            {"name": "bud", "attribute": "pet_count", "value": 12},
            {"name": "pat", "attribute": "pet_count", "value": 0},
        ],
        system_settings: None,
    }
    """

    entity_settings: list[EntitySettingModel]
    system_settings: Optional[list[SystemSettingModel]] = None

    @validator("system_settings")
    def lower_case_name(cls, v):
        if v is None:
            raise NotImplementedError("system_setting is not implemented yet")
        return v


class QueryModel(BaseModel):
    """translate a question about the story outcome into a programatic expression"""

    question: str = Field(alias=Constant.narrative_input.value)  # input
    expression: str  # output, part of llm completion
    llm_error_msg: str  # output, part of llm completion
    _result_table: str = PrivateAttr()  # result of the executed query


class StoryModel(BaseModel):
    """the output of calling a StoryChain instance"""

    causal_mental_model: Any = Field(required=True)  # story's plot
    intervention: Any = Field(required=True)  # reset of story initial conditions
    query: Any = Field(required=True)  # question about the story outcome
    _outcome_table: pd.DataFrame = PrivateAttr(default=None)
    _DAG: nx.DiGraph = PrivateAttr(default=None)

    """
    @root_validator
    def validate_subclass(cls, values):
        required_base_models = {
            "causal_mental_model": CausalModel,
            "intervention": InterventionModel,
            .query": QueryModel,
        }
        for field, required_base_model in required_base_models.items():
            given_pydantic_object = values[field]
            if issubclass(
                type(given_pydantic_object), type(required_base_model)
            ) or isinstance(given_pydantic_object, type(required_base_model)):
                return values
            else:
                raise TypeError(
                    f"Wrong type for '{field}', "
                    f"must be subclass of '{required_base_model}'"
                )
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compute()

        # TODO: replace with  `__post_init__` when pydantic v2 is adopted
        # This overriding of init results in missing constructor parameter hints
        # https://github.com/pydantic/pydantic/issues/1729#issuecomment-1300576214

    @root_validator
    def check_intervention_is_valid(cls, values):
        valid_names = [e.name for e in values["causal_mental_model"].entities]
        for setting in values["intervention"].entity_settings:
            if setting.name not in valid_names:
                error_msg = f"""
                    Hypothetical question has an invalid entity name.
                    `{setting.name}` not in `{valid_names}`
                """
                raise ValueError(error_msg)
        return values

    def _block_back_door_paths(self) -> None:
        # stop intervention entities from depending on others
        intervention_entities = [
            entity_setting.name for entity_setting in self.intervention.entity_settings
        ]
        for entity in self.causal_mental_model.entities:
            if entity.name in intervention_entities:
                entity.depends_on = []
                entity.code = "pass"

    def _set_initial_conditions(self) -> None:
        for entity_setting in self.intervention.entity_settings:
            for entity in self.causal_mental_model.entities:
                if entity.name == entity_setting.name:
                    entity.value = entity_setting.value

    def _make_graph(self) -> None:
        self._DAG = nx.DiGraph()
        for entity in self.causal_mental_model.entities:
            for parent_name in entity.depends_on:
                self._DAG.add_edge(parent_name, entity.name)

        # entities that have no interdependent relations are dropped
        self.causal_mental_model.entities = [
            entity
            for entity in self.causal_mental_model.entities
            if entity.name in self._DAG.nodes
        ]

    def _sort_entities(self) -> None:
        # order the sequence of causal actions
        sorted_nodes = list(nx.topological_sort(self._DAG))
        self.causal_mental_model.entities.sort(key=lambda x: sorted_nodes.index(x.name))

    def _forward_propagate(self) -> None:
        entity_scope = {
            entity.name: entity for entity in self.causal_mental_model.entities
        }
        for entity in self.causal_mental_model.entities:
            if entity.code == "pass":
                continue
            else:
                # gist.github.com/dean0x7d/df5ce97e4a1a05be4d56d1378726ff92
                exec(entity.code, globals(), entity_scope)
        row_values = [entity.dict() for entity in entity_scope.values()]
        self._outcome_table = pd.DataFrame(row_values)

    def _run_query(self) -> None:
        def humanize_sql_error_msg(error):
            pattern = r"column\s+(.*?)\s+not found"
            col_match = re.search(pattern, error)
            if col_match:
                return (
                    "SQL error: "
                    + col_match.group(1)
                    + " is not an attribute in your story!"
                )
            else:
                return str(error)

        if self.query.llm_error_msg == "":
            try:
                df = self._outcome_table  # noqa
                query_result = duckdb.sql(self.query.expression).df()
                self.query._result_table = query_result
            except duckdb.BinderException as e:
                self.query._result_table = humanize_sql_error_msg(str(e))
            except Exception as e:
                self.query._result_table = str(e)
        else:
            msg = "LLM maybe failed to translate question to SQL query."
            raise ValueError(
                {
                    "question": self.query.question,
                    "llm_error_msg": self.query.llm_error_msg,
                    "msg": msg,
                }
            )

    def _compute(self) -> Any:
        self._block_back_door_paths()
        self._set_initial_conditions()
        self._make_graph()
        self._sort_entities()
        self._forward_propagate()
        self._run_query()

    def print_debug_report(self) -> None:
        report = {
            "outcome": self._outcome_table,
            "query": self.query.dict(),
            "result": self.query._result_table,
        }
        from pprint import pprint

        pprint(report)
