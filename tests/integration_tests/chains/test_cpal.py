"""Test CPAL chain."""
import os
import unittest
import pytest
from dotenv import load_dotenv
from langchain.experimental.chains.cpal.base import (
    CPALChain,
    CausalChain,
    InterventionChain,
    QueryChain,
    NarrativeChain,
)
from langchain.experimental.chains.cpal.models import (
    CausalModel,
    InterventionModel,
    NarrativeModel,
    QueryModel,
    EntityModel,
)
from langchain.experimental.chains.cpal.constants import Constant

import pygraphviz

# allow local env to override default LLM client setting
load_dotenv()

default_completion_config = dict(temperature=0, max_tokens=512)  # type: ignore

if os.getenv("OPENAI_API_TYPE") == "azure":
    from langchain.llms import AzureOpenAI

    local_azure_config = dict(deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"))
    llm_config = {**default_completion_config, **local_azure_config}
    llm_client = AzureOpenAI
else:
    from langchain import OpenAI

    llm_config = {**default_completion_config}
    llm_client = OpenAI

llm = llm_client(**llm_config)  # type: ignore


class TestCPALChain_MathWordProblems(unittest.TestCase):
    """Test the CPAL chain and its component chains on math word problems."""

    def test_causal_chain(self) -> None:
        """Test CausalChain can translate a narrative's plot into a causal model
        containing operations linked by a DAG."""

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
            "narrative_input": "Jan has three times the number of pets as Marcia. Marcia "
            "has two more pets than Cindy. ",
        }
        self.assertDictEqual(output, expected_output)
        self.assertEqual(
            isinstance(output[Constant.chain_data.value], CausalModel), True
        )

    def test_intervention_chain(self) -> None:
        """Test InterventionChain translates a hypothetical into a new value setting."""

        story_conditions_chain = InterventionChain.from_univariate_prompt(llm)
        question = "if cindy has ten pets"
        data = story_conditions_chain(question)[Constant.chain_data.value]
        self.assertEqual(type(data), InterventionModel)

    def test_intervention_chain_2(self) -> None:
        """Test InterventionChain translates a hypothetical into a new value setting."""

        story_conditions_chain = InterventionChain.from_univariate_prompt(llm)
        narrative_condition = "What if Cindy has ten pets and Boris has 5 pets? "
        data = story_conditions_chain(narrative_condition)[Constant.chain_data.value]
        self.assertEqual(type(data), InterventionModel)

    def test_query_chain(self) -> None:
        """Test QueryChain translates a question into a query expression."""
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
        narrative_chain = NarrativeChain.from_univariate_prompt(llm)
        data = narrative_chain(narrative)[Constant.chain_data.value]
        self.assertEqual(type(data), NarrativeModel)

        out = narrative_chain(narrative)
        expected_narrative_out = {
            "chain_answer": None,
            "chain_data": NarrativeModel(
                story_outcome_question="how many pets does Jan have?",
                story_hypothetical="If Cindy has ten pets",
                story_plot="Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy.",
            ),
            "narrative_input": "Jan has three times the number of pets as Marcia. Marcia "
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
            "Jan has three times the number of pets as Marcia."
            "Marcia has two more pets than Cindy."
            "If Cindy has ten pets, how many pets does Jan have?"
        )
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        output = cpal_chain(narrative_input)
        data = output[Constant.chain_data.value]

        expected_output = {
            "causal_mental_model": {
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

        # TODO: replace below with a CPAL Report chain
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
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        with pytest.raises(Exception) as e_info:
            print(e_info)
            cpal_chain.run(narrative_input)

    def test_causal_mediator(self) -> None:
        """
        Test CPAL approach on causal mediator.

        We bypass Cindy and give pets straight to Marcia.
        """

        narrative_input = (
            "Jan has three times the number of pets as Marcia."
            "Marcia has two more pets than Cindy."
            "If Marcia has ten pets, how many pets does Jan have?"
        )
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        answer = cpal_chain.run(narrative_input)
        self.assertEqual(answer, 30.0)

    @pytest.mark.skip(reason="requires `sudo apt-get install graphviz-dev`")
    @pytest.mark.requires(pygraphviz)
    def test_graph(self) -> None:
        """
        Test CPAL makes a graphviz diagram.

        Catch two types of import errors

        - if graphviz is not installed
        - if pygraphviz is not installed
        """

        narrative_input = (
            "Jan has three times the number of pets as Marcia."
            "Marcia has two more pets than Cindy."
            "If Marcia has ten pets, how many pets does Jan have?"
        )
        cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
        cpal_chain.run(narrative_input)
        graph_diagram = cpal_chain.graph()

        self.assertEqual(isinstance(graph_diagram, pygraphviz.agraph.AGraph), True)
