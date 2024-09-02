from __future__ import annotations  # allows pydantic model to reference itself

import re
from typing import Any, List, Optional, Union

from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

from langchain_experimental.cpal.constants import Constant
from langchain_experimental.pydantic_v1 import (
    BaseModel,
    Field,
    PrivateAttr,
    root_validator,
    validator,
)


class NarrativeModel(BaseModel):
    """
    Narrative input as three story elements.
    """

    story_outcome_question: str
    story_hypothetical: str
    story_plot: str  # causal stack of operations

    @validator("*", pre=True)
    def empty_str_to_none(cls, v: str) -> Union[str, None]:
        """Empty strings are not allowed"""
        if v == "":
            return None
        return v


class EntityModel(BaseModel):
    """Entity in the story."""

    name: str = Field(description="entity name")
    code: str = Field(description="entity actions")
    value: float = Field(description="entity initial value")
    depends_on: List[str] = Field(default=[], description="ancestor entities")

    # TODO: generalize to multivariate math
    # TODO: acyclic graph

    class Config:
        validate_assignment = True

    @validator("name")
    def lower_case_name(cls, v: str) -> str:
        v = v.lower()
        return v


class CausalModel(BaseModel):
    """Casual data."""

    attribute: str = Field(description="name of the attribute to be calculated")
    entities: List[EntityModel] = Field(description="entities in the story")

    # TODO: root validate each `entity.depends_on` using system's entity names


class EntitySettingModel(BaseModel):
    """Entity initial conditions.

    Initial conditions for an entity

    {"name": "bud", "attribute": "pet_count", "value": 12}
    """

    name: str = Field(description="name of the entity")
    attribute: str = Field(description="name of the attribute to be calculated")
    value: float = Field(description="entity's attribute value (calculated)")

    @validator("name")
    def lower_case_transform(cls, v: str) -> str:
        v = v.lower()
        return v


class SystemSettingModel(BaseModel):
    """System initial conditions.

    Initial global conditions for the system.

    {"parameter": "interest_rate", "value": .05}
    """

    parameter: str
    value: float


class InterventionModel(BaseModel):
    """Intervention data of the story aka initial conditions.

    >>> intervention.dict()
    {
        entity_settings: [
            {"name": "bud", "attribute": "pet_count", "value": 12},
            {"name": "pat", "attribute": "pet_count", "value": 0},
        ],
        system_settings: None,
    }
    """

    entity_settings: List[EntitySettingModel]
    system_settings: Optional[List[SystemSettingModel]] = None

    @validator("system_settings")
    def lower_case_name(cls, v: str) -> Union[str, None]:
        if v is not None:
            raise NotImplementedError("system_setting is not implemented yet")
        return v


class QueryModel(BaseModel):
    """Query data of the story.

    translate a question about the story outcome into a programmatic expression"""

    question: str = Field(  # type: ignore[literal-required]
        alias=Constant.narrative_input.value
    )  # input  # type: ignore[literal-required]
    expression: str  # output, part of llm completion
    llm_error_msg: str  # output, part of llm completion
    _result_table: str = PrivateAttr()  # result of the executed query


class ResultModel(BaseModel):
    """Result of the story query."""

    question: str = Field(  # type: ignore[literal-required]
        alias=Constant.narrative_input.value
    )  # input  # type: ignore[literal-required]
    _result_table: str = PrivateAttr()  # result of the executed query


class StoryModel(BaseModel):
    """Story data."""

    causal_operations: Any = Field()
    intervention: Any = Field()
    query: Any = Field()
    _outcome_table: Any = PrivateAttr(default=None)
    _networkx_wrapper: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._compute()

        # TODO: when langchain adopts pydantic.v2 replace w/ `__post_init__`
        # misses hints github.com/pydantic/pydantic/issues/1729#issuecomment-1300576214

    # TODO: move away from `root_validator` since it is deprecated in pydantic v2
    #       and causes mypy type-checking failures (hence the `type: ignore`)
    @root_validator  # type: ignore[call-overload]
    def check_intervention_is_valid(cls, values: dict) -> dict:
        valid_names = [e.name for e in values["causal_operations"].entities]
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
        for entity in self.causal_operations.entities:
            if entity.name in intervention_entities:
                entity.depends_on = []
                entity.code = "pass"

    def _set_initial_conditions(self) -> None:
        for entity_setting in self.intervention.entity_settings:
            for entity in self.causal_operations.entities:
                if entity.name == entity_setting.name:
                    entity.value = entity_setting.value

    def _make_graph(self) -> None:
        self._networkx_wrapper = NetworkxEntityGraph()
        for entity in self.causal_operations.entities:
            for parent_name in entity.depends_on:
                self._networkx_wrapper._graph.add_edge(
                    parent_name, entity.name, relation=entity.code
                )

        # TODO: is it correct to drop entities with no impact on the outcome (?)
        self.causal_operations.entities = [
            entity
            for entity in self.causal_operations.entities
            if entity.name in self._networkx_wrapper.get_topological_sort()
        ]

    def _sort_entities(self) -> None:
        # order the sequence of causal actions
        sorted_nodes = self._networkx_wrapper.get_topological_sort()
        self.causal_operations.entities.sort(key=lambda x: sorted_nodes.index(x.name))

    def _forward_propagate(self) -> None:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Unable to import pandas, please install with `pip install pandas`."
            ) from e
        entity_scope = {
            entity.name: entity for entity in self.causal_operations.entities
        }
        for entity in self.causal_operations.entities:
            if entity.code == "pass":
                continue
            else:
                # gist.github.com/dean0x7d/df5ce97e4a1a05be4d56d1378726ff92
                exec(entity.code, globals(), entity_scope)
        row_values = [entity.dict() for entity in entity_scope.values()]
        self._outcome_table = pd.DataFrame(row_values)

    def _run_query(self) -> None:
        def humanize_sql_error_msg(error: str) -> str:
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
                import duckdb

                df = self._outcome_table  # noqa
                query_result = duckdb.sql(self.query.expression).df()
                self.query._result_table = query_result
            except duckdb.BinderException as e:
                self.query._result_table = humanize_sql_error_msg(str(e))
            except ImportError as e:
                raise ImportError(
                    "Unable to import duckdb, please install with `pip install duckdb`."
                ) from e
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
