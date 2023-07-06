"""Test CPAL chain and its chain components on simple univariate math."""
import json
import pydantic
import pytest
from unittest import mock

from tests.unit_tests.llms.fake_llm import FakeLLM
from typing import Type

from langchain.experimental.chains.cpal.base import (
    CPALChain,
    NarrativeChain,
    CausalChain,
    InterventionChain,
    QueryChain,
    make_prompt_template,
)
from langchain.experimental.chains.cpal.models import (
    NarrativeModel,
    CausalModel,
    QueryModel,
    InterventionModel,
    EntityModel,
    EntitySettingModel,
)

from langchain.experimental.chains.cpal.templates.univariate.narrative import (
    template as narrative_template,
)

from langchain.experimental.chains.cpal.templates.univariate.causal import (
    template as causal_template,
)

from langchain.experimental.chains.cpal.templates.univariate.intervention import (
    template as intervention_template,
)

from langchain.experimental.chains.cpal.templates.univariate.query import (
    template as query_template,
)


class MockData(pydantic.BaseModel):
    """Mock prompt and completion data to hydrate out fake LLM service."""

    question: str
    completion: str
    template: str
    data_model: Type[pydantic.BaseModel]

    @property
    def prompt(self) -> str:
        """Format the prompt with the question."""
        template = make_prompt_template(
            template=self.template, data_model=self.data_model
        )
        prompt = template.format(narrative_input=self.question)
        return prompt


@pytest.fixture
def fake_llm() -> FakeLLM:
    """
    Fake LLM service for testing CPAL chain and its components chains
    on univariate math examples.
    """
    narrative = MockData(
        **{
            "question": (
                "jan has three times the number of pets as marcia. "
                "marcia has two more pets than cindy."
                "if cindy has ten pets, how many pets does jan have? "
            ),
            "completion": json.dumps(
                {
                    "story_outcome_question": "how many pets does jan have? ",
                    "story_hypothetical": "if cindy has ten pets",
                    "story_plot": "jan has three times the number of pets as marcia. marcia has two more pets than cindy.",  # noqa: E501
                }
            ),
            "template": narrative_template,
            "data_model": NarrativeModel,
        }
    )

    causal_model = MockData(
        **{
            "question": (
                "jan has three times the number of pets as marcia. "
                "marcia has two more pets than cindy."
            ),
            "completion": (
                "\n"
                "{\n"
                '    "attribute": "pet_count",\n'
                '    "entities": [\n'
                "        {\n"
                '            "name": "cindy",\n'
                '            "value": 0,\n'
                '            "depends_on": [],\n'
                '            "code": "pass"\n'
                "        },\n"
                "        {\n"
                '            "name": "marcia",\n'
                '            "value": 0,\n'
                '            "depends_on": ["cindy"],\n'
                '            "code": "marcia.value = cindy.value + 2"\n'
                "        },\n"
                "        {\n"
                '            "name": "jan",\n'
                '            "value": 0,\n'
                '            "depends_on": ["marcia"],\n'
                '            "code": "jan.value = marcia.value * 3"\n'
                "        }\n"
                "    ]\n"
                "}"
            ),
            "template": causal_template,
            "data_model": CausalModel,
        }
    )

    intervention = MockData(
        **{
            "question": ("if cindy has ten pets"),
            "completion": (
                "{\n"
                '    "entity_settings" : [\n'
                '        { "name": "cindy", "attribute": "pet_count", "value": "10" }\n'
                "    ]\n"
                "}"
            ),
            "template": intervention_template,
            "data_model": InterventionModel,
        }
    )

    query = MockData(
        **{
            "question": ("how many pets does jan have? "),
            "completion": (
                "{\n"
                '    "narrative_input": "how many pets does jan have? ",\n'
                '    "llm_error_msg": "",\n'
                '    "expression": "SELECT name, value FROM df WHERE name = \'jan\'"\n'
                "}"
            ),
            "template": query_template,
            "data_model": QueryModel,
        }
    )

    fake_llm = FakeLLM()
    fake_llm.queries = {}
    for mock_data in [narrative, causal_model, intervention, query]:
        fake_llm.queries.update({mock_data.prompt: mock_data.completion})
    return fake_llm


def test_narrative_chain(fake_llm) -> None:
    """Test narrative chain returns the three main elements of the causal
    narrative as a pydantic object.
    """
    narrative_chain = NarrativeChain.from_univariate_prompt(llm=fake_llm)
    output = narrative_chain(
        (
            "jan has three times the number of pets as marcia. "
            "marcia has two more pets than cindy."
            "if cindy has ten pets, how many pets does jan have? "
        )
    )
    expected_output = {
        "chain_answer": None,
        "chain_data": NarrativeModel(
            story_outcome_question="how many pets does jan have? ",
            story_hypothetical="if cindy has ten pets",
            story_plot="jan has three times the number of pets as marcia. marcia has two more pets than cindy.",  # noqa: E501
        ),
        "narrative_input": "jan has three times the number of pets as marcia. marcia "
        "has two more pets than cindy.if cindy has ten pets, how "
        "many pets does jan have? ",
    }
    assert output == expected_output


def test_causal_chain(fake_llm) -> None:
    """
    Test causal chain returns a DAG as a pydantic object.
    """
    causal_chain = CausalChain.from_univariate_prompt(llm=fake_llm)
    output = causal_chain(
        (
            "jan has three times the number of pets as "
            "marcia. marcia has two more pets than cindy."
        )
    )
    expected_output = {
        "chain_answer": None,
        "chain_data": CausalModel(
            attribute="pet_count",
            entities=[
                EntityModel(name="cindy", code="pass", value=0.0, depends_on=[]),
                EntityModel(
                    name="marcia",
                    code="marcia.value = cindy.value + 2",
                    value=0.0,
                    depends_on=["cindy"],
                ),
                EntityModel(
                    name="jan",
                    code="jan.value = marcia.value * 3",
                    value=0.0,
                    depends_on=["marcia"],
                ),
            ],
        ),
        "narrative_input": "jan has three times the number of pets as marcia. marcia "
        "has two more pets than cindy.",
    }
    assert output == expected_output


def test_intervention_chain(fake_llm) -> None:
    """
    Test intervention chain correctly transforms
    the LLM's text completion into a setting-like object.
    """
    intervention_chain = InterventionChain.from_univariate_prompt(llm=fake_llm)
    output = intervention_chain("if cindy has ten pets")
    expected_output = {
        "chain_answer": None,
        "chain_data": InterventionModel(
            entity_settings=[
                EntitySettingModel(name="cindy", attribute="pet_count", value=10),
            ]
        ),
        "narrative_input": "if cindy has ten pets",
    }
    assert output == expected_output


def test_query_chain(fake_llm) -> None:
    """
    Test query chain correctly transforms
    the LLM's text completion into a query-like object.
    """
    query_chain = QueryChain.from_univariate_prompt(llm=fake_llm)
    output = query_chain("how many pets does jan have? ")
    expected_output = {
        "chain_answer": None,
        "chain_data": QueryModel(
            narrative_input="how many pets does jan have? ",
            llm_error_msg="",
            expression="SELECT name, value FROM df WHERE name = 'jan'",
        ),
        "narrative_input": "how many pets does jan have? ",
    }
    assert output == expected_output


def test_cpal_chain(fake_llm) -> None:
    """
    patch required since `networkx` package is not part of unit test environment
    """
    with mock.patch(
        "langchain.experimental.chains.cpal.models.NetworkxEntityGraph"
    ) as mock_networkx:
        graph_instance = mock_networkx.return_value
        graph_instance.get_topological_sort.return_value = ["cindy", "marcia", "jan"]
        cpal_chain = CPALChain.from_univariate_prompt(llm=fake_llm, verbose=True)
        cpal_chain.run(
            (
                "jan has three times the number of pets as "
                "marcia. marcia has two more pets than cindy."
                "if cindy has ten pets, how many pets does jan have? "
            )
        )
