"""Test CPAL chain."""

import json
import unittest
from typing import Type
from unittest import mock

import pytest
from langchain import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate

from langchain_experimental import pydantic_v1 as pydantic
from langchain_experimental.cpal.base import (
    CausalChain,
    CPALChain,
    InterventionChain,
    NarrativeChain,
    QueryChain,
)
from langchain_experimental.cpal.constants import Constant
from langchain_experimental.cpal.models import (
    CausalModel,
    EntityModel,
    EntitySettingModel,
    InterventionModel,
    NarrativeModel,
    QueryModel,
)
from langchain_experimental.cpal.templates.univariate.causal import (
    template as causal_template,
)
from langchain_experimental.cpal.templates.univariate.intervention import (
    template as intervention_template,
)
from langchain_experimental.cpal.templates.univariate.narrative import (
    template as narrative_template,
)
from langchain_experimental.cpal.templates.univariate.query import (
    template as query_template,
)
from tests.unit_tests.llms.fake_llm import FakeLLM


class TestUnitCPALChain_MathWordProblems(unittest.TestCase):
    """Unit Test the CPAL chain and its component chains on math word problems.

    These tests can't run in the standard unit test directory because of
    this issue, https://github.com/hwchase17/langchain/issues/7451

    """

    def setUp(self) -> None:
        self.fake_llm = self.make_fake_llm()

    def make_fake_llm(self) -> FakeLLM:
        """
        Fake LLM service for testing CPAL chain and its components chains
        on univariate math examples.
        """

        class LLMMockData(pydantic.BaseModel):
            question: str
            completion: str
            template: str
            data_model: Type[pydantic.BaseModel]

            @property
            def prompt(self) -> str:
                """Create LLM prompt with the question."""
                prompt_template = PromptTemplate(
                    input_variables=[Constant.narrative_input.value],
                    template=self.template,
                    partial_variables={
                        "format_instructions": PydanticOutputParser(
                            pydantic_object=self.data_model
                        ).get_format_instructions()
                    },
                )
                prompt = prompt_template.format(narrative_input=self.question)
                return prompt

        narrative = LLMMockData(
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

        causal_model = LLMMockData(
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

        intervention = LLMMockData(
            **{
                "question": ("if cindy has ten pets"),
                "completion": (
                    "{\n"
                    '    "entity_settings" : [\n'
                    '        { "name": "cindy", "attribute": "pet_count", "value": "10" }\n'  # noqa: E501
                    "    ]\n"
                    "}"
                ),
                "template": intervention_template,
                "data_model": InterventionModel,
            }
        )

        query = LLMMockData(
            **{
                "question": ("how many pets does jan have? "),
                "completion": (
                    "{\n"
                    '    "narrative_input": "how many pets does jan have? ",\n'
                    '    "llm_error_msg": "",\n'
                    '    "expression": "SELECT name, value FROM df WHERE name = \'jan\'"\n'  # noqa: E501
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

    def test_narrative_chain(self) -> None:
        """Test narrative chain returns the three main elements of the causal
        narrative as a pydantic object.
        """
        narrative_chain = NarrativeChain.from_univariate_prompt(llm=self.fake_llm)
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
            "narrative_input": "jan has three times the number of pets as marcia. marcia "  # noqa: E501
            "has two more pets than cindy.if cindy has ten pets, how "
            "many pets does jan have? ",
        }
        assert output == expected_output

    def test_causal_chain(self) -> None:
        """
        Test causal chain returns a DAG as a pydantic object.
        """
        causal_chain = CausalChain.from_univariate_prompt(llm=self.fake_llm)
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
            "narrative_input": "jan has three times the number of pets as marcia. marcia "  # noqa: E501
            "has two more pets than cindy.",
        }
        assert output == expected_output

    def test_intervention_chain(self) -> None:
        """
        Test intervention chain correctly transforms
        the LLM's text completion into a setting-like object.
        """
        intervention_chain = InterventionChain.from_univariate_prompt(llm=self.fake_llm)
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

    def test_query_chain(self) -> None:
        """
        Test query chain correctly transforms
        the LLM's text completion into a query-like object.
        """
        query_chain = QueryChain.from_univariate_prompt(llm=self.fake_llm)
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

    def test_cpal_chain(self) -> None:
        """
        patch required since `networkx` package is not part of unit test environment
        """
        with mock.patch(
            "langchain_experimental.cpal.models.NetworkxEntityGraph"
        ) as mock_networkx:
            graph_instance = mock_networkx.return_value
            graph_instance.get_topological_sort.return_value = [
                "cindy",
                "marcia",
                "jan",
            ]
            cpal_chain = CPALChain.from_univariate_prompt(
                llm=self.fake_llm, verbose=True
            )
            cpal_chain.run(
                (
                    "jan has three times the number of pets as "
                    "marcia. marcia has two more pets than cindy."
                    "if cindy has ten pets, how many pets does jan have? "
                )
            )


class TestCPALChain_MathWordProblems(unittest.TestCase):
    """Test the CPAL chain and its component chains on math word problems."""

    def test_causal_chain(self) -> None:
        """Test CausalChain can translate a narrative's plot into a causal model
        containing operations linked by a DAG."""

        llm = OpenAI(temperature=0, max_tokens=512)
        casual_chain = CausalChain.from_univariate_prompt(llm)
        narrative_plot = (
            "Jan has three times the number of pets as Marcia. "
            "Marcia has two more pets than Cindy. "
        )
        output = casual_chain(narrative_plot)
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
            "narrative_input": "Jan has three times the number of pets as Marcia. Marcia "  # noqa: E501
            "has two more pets than Cindy. ",
        }
        self.assertDictEqual(output, expected_output)
        self.assertEqual(
            isinstance(output[Constant.chain_data.value], CausalModel), True
        )

    def test_intervention_chain(self) -> None:
        """Test InterventionChain translates a hypothetical into a new value setting."""

        llm = OpenAI(temperature=0, max_tokens=512)
        story_conditions_chain = InterventionChain.from_univariate_prompt(llm)
        question = "if cindy has ten pets"
        data = story_conditions_chain(question)[Constant.chain_data.value]
        self.assertEqual(type(data), InterventionModel)

    def test_intervention_chain_2(self) -> None:
        """Test InterventionChain translates a hypothetical into a new value setting."""

        llm = OpenAI(temperature=0, max_tokens=512)
        story_conditions_chain = InterventionChain.from_univariate_prompt(llm)
        narrative_condition = "What if Cindy has ten pets and Boris has 5 pets? "
        data = story_conditions_chain(narrative_condition)[Constant.chain_data.value]
        self.assertEqual(type(data), InterventionModel)

    def test_query_chain(self) -> None:
        """Test QueryChain translates a question into a query expression."""
        llm = OpenAI(temperature=0, max_tokens=512)
        query_chain = QueryChain.from_univariate_prompt(llm)
        narrative_question = "How many pets will Marcia end up with? "
        data = query_chain(narrative_question)[Constant.chain_data.value]
        self.assertEqual(type(data), QueryModel)

    def test_narrative_chain(self) -> None:
        """Test NarrativeChain decomposes a human's narrative into three story elements:

        - causal model
        - intervention model
        - query model
        """

        narrative = (
            "Jan has three times the number of pets as Marcia. "
            "Marcia has two more pets than Cindy. "
            "If Cindy has ten pets, how many pets does Jan have? "
        )
        llm = OpenAI(temperature=0, max_tokens=512)
        narrative_chain = NarrativeChain.from_univariate_prompt(llm)
        data = narrative_chain(narrative)[Constant.chain_data.value]
        self.assertEqual(type(data), NarrativeModel)

        out = narrative_chain(narrative)
        expected_narrative_out = {
            "chain_answer": None,
            "chain_data": NarrativeModel(
                story_outcome_question="how many pets does Jan have?",
                story_hypothetical="If Cindy has ten pets",
                story_plot="Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy.",  # noqa: E501
            ),
            "narrative_input": "Jan has three times the number of pets as Marcia. Marcia "  # noqa: E501
            "has two more pets than Cindy. If Cindy has ten pets, how "
            "many pets does Jan have? ",
        }
        self.assertDictEqual(out, expected_narrative_out)

    def test_against_pal_chain_doc(self) -> None:
        """
        Test CPAL chain against the first example in the PAL chain notebook doc:

        https://github.com/hwchase17/langchain/blob/master/docs/extras/modules/chains/additional/pal.ipynb
        """

        narrative_input = (
            "Jan has three times the number of pets as Marcia."
            " Marcia has two more pets than Cindy."
            " If Cindy has four pets, how many total pets do the three have?"
        )

        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        answer = cpal_chain.run(narrative_input)

        """
        >>> story._outcome_table
             name                            code  value depends_on
        0   cindy                            pass    4.0         []
        1  marcia  marcia.value = cindy.value + 2    6.0    [cindy]
        2     jan    jan.value = marcia.value * 3   18.0   [marcia]

        """
        self.assertEqual(answer, 28.0)

    def test_simple(self) -> None:
        """
        Given a simple math word problem here we are test and illustrate the
        the data structures that are produced by the CPAL chain.
        """

        narrative_input = (
            "jan has three times the number of pets as marcia."
            "marcia has two more pets than cindy."
            "If cindy has ten pets, how many pets does jan have?"
        )
        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        output = cpal_chain(narrative_input)
        data = output[Constant.chain_data.value]

        expected_output = {
            "causal_operations": {
                "attribute": "pet_count",
                "entities": [
                    {"code": "pass", "depends_on": [], "name": "cindy", "value": 10.0},
                    {
                        "code": "marcia.value = cindy.value + 2",
                        "depends_on": ["cindy"],
                        "name": "marcia",
                        "value": 12.0,
                    },
                    {
                        "code": "jan.value = marcia.value * 3",
                        "depends_on": ["marcia"],
                        "name": "jan",
                        "value": 36.0,
                    },
                ],
            },
            "intervention": {
                "entity_settings": [
                    {"attribute": "pet_count", "name": "cindy", "value": 10.0}
                ],
                "system_settings": None,
            },
            "query": {
                "expression": "SELECT name, value FROM df WHERE name = 'jan'",
                "llm_error_msg": "",
                "question": "how many pets does jan have?",
            },
        }
        self.assertDictEqual(data.dict(), expected_output)

        """
        Illustrate the query model's result table as a printed pandas dataframe
        >>> data._outcome_table
             name                            code  value depends_on
        0   cindy                            pass   10.0         []
        1  marcia  marcia.value = cindy.value + 2   12.0    [cindy]
        2     jan    jan.value = marcia.value * 3   36.0   [marcia]
        """

        expected_output = {
            "code": {
                0: "pass",
                1: "marcia.value = cindy.value + 2",
                2: "jan.value = marcia.value * 3",
            },
            "depends_on": {0: [], 1: ["cindy"], 2: ["marcia"]},
            "name": {0: "cindy", 1: "marcia", 2: "jan"},
            "value": {0: 10.0, 1: 12.0, 2: 36.0},
        }
        self.assertDictEqual(data._outcome_table.to_dict(), expected_output)

        expected_output = {"name": {0: "jan"}, "value": {0: 36.0}}
        self.assertDictEqual(data.query._result_table.to_dict(), expected_output)

        # TODO: use an LLM chain to translate numbers to words
        df = data.query._result_table
        expr = "name == 'jan'"
        answer = df.query(expr).iloc[0]["value"]
        self.assertEqual(float(answer), 36.0)

    def test_hallucinating(self) -> None:
        """
        Test CPAL approach does not hallucinate when given
        an invalid entity in the question.

        The PAL chain would hallucinates here!
        """

        narrative_input = (
            "Jan has three times the number of pets as Marcia."
            "Marcia has two more pets than Cindy."
            "If Cindy has ten pets, how many pets does Barak have?"
        )
        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        with pytest.raises(Exception) as e_info:
            print(e_info)
            cpal_chain.run(narrative_input)

    def test_causal_mediator(self) -> None:
        """
        Test CPAL approach on causal mediator.
        """

        narrative_input = (
            "jan has three times the number of pets as marcia."
            "marcia has two more pets than cindy."
            "If marcia has ten pets, how many pets does jan have?"
        )
        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        answer = cpal_chain.run(narrative_input)
        self.assertEqual(answer, 30.0)

    @pytest.mark.skip(reason="requires manual install of debian and py packages")
    def test_draw(self) -> None:
        """
        Test CPAL chain can draw its resulting DAG.
        """
        import os

        narrative_input = (
            "Jan has three times the number of pets as Marcia."
            "Marcia has two more pets than Cindy."
            "If Marcia has ten pets, how many pets does Jan have?"
        )
        llm = OpenAI(temperature=0, max_tokens=512)
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        cpal_chain.run(narrative_input)
        path = "graph.svg"
        cpal_chain.draw(path=path)
        self.assertTrue(os.path.exists(path))
