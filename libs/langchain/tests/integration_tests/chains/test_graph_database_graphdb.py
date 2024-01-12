from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest

from langchain.chains import LLMChain
from langchain.chains.graph_qa.graphdb import GraphDBQAChain
from langchain.chat_models import ChatOpenAI
from langchain.graphs import GraphDBGraph

"""
cd libs/langchain/tests/integration_tests/graphs/graphdb
./start.sh
"""


@pytest.mark.parametrize('max_regeneration_attempts', [-2, -1, 0, 1, 2])
def test_valid_sparql(max_regeneration_attempts: int) -> None:
    graph = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.ttl')
    )
    chain = GraphDBQAChain.from_llm(
        Mock(ChatOpenAI), graph=graph, max_regeneration_attempts=max_regeneration_attempts
    )
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_regeneration_select_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_select_chain.run = MagicMock(return_value='SELECT * {?s ?p ?o} LIMIT 1')
    chain.sparql_regeneration_select_chain.run = MagicMock()
    chain.qa_chain = MagicMock(return_value={
        'text': 'Tatooine',
        'context': [],
        'prompt': 'What is Luke Skywalker\'s home planet?'
    })
    chain.qa_chain.output_key = 'text'

    result = chain.run('What is Luke Skywalker\'s home planet?')

    assert chain.sparql_generation_select_chain.run.call_count == 1
    assert chain.sparql_regeneration_select_chain.run.call_count == 0
    assert chain.qa_chain.call_count == 1
    assert result == 'Tatooine'


@pytest.mark.parametrize('max_regeneration_attempts', [-2, -1, 0])
def test_invalid_sparql_non_positive_max_regeneration_attempts(max_regeneration_attempts: int) -> None:
    graph = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.ttl')
    )
    chain = GraphDBQAChain.from_llm(
        Mock(ChatOpenAI), graph=graph, max_regeneration_attempts=max_regeneration_attempts
    )
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_regeneration_select_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_select_chain.run = MagicMock(return_value='```sparql SELECT * {?s ?p ?o} LIMIT 1```')
    chain.sparql_regeneration_select_chain.run = MagicMock()

    with pytest.raises(ValueError) as e:
        chain.run('What is Luke Skywalker\'s home planet?')

    assert str(e.value) == 'The generated SPARQL query is invalid.'

    assert chain.sparql_generation_select_chain.run.call_count == 1
    assert chain.sparql_regeneration_select_chain.run.call_count == 0
    assert chain.qa_chain.call_count == 0


@pytest.mark.parametrize('max_regeneration_attempts', [1, 2, 3])
def test_valid_sparql_after_first_retry(max_regeneration_attempts: int) -> None:
    graph = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.ttl')
    )
    chain = GraphDBQAChain.from_llm(
        Mock(ChatOpenAI), graph=graph, max_regeneration_attempts=max_regeneration_attempts
    )
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_regeneration_select_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_select_chain.run = MagicMock(return_value='```sparql SELECT * {?s ?p ?o} LIMIT 1```')
    chain.sparql_regeneration_select_chain.run = MagicMock(return_value='SELECT * {?s ?p ?o} LIMIT 1')
    chain.qa_chain = MagicMock(return_value={
        'text': 'Tatooine',
        'context': [],
        'prompt': 'What is Luke Skywalker\'s home planet?'
    })
    chain.qa_chain.output_key = 'text'

    result = chain.run('What is Luke Skywalker\'s home planet?')

    assert chain.sparql_generation_select_chain.run.call_count == 1
    assert chain.sparql_regeneration_select_chain.run.call_count == 1
    assert chain.qa_chain.call_count == 1
    assert result == 'Tatooine'


@pytest.mark.parametrize('max_regeneration_attempts', [1, 2, 3])
def test_invalid_sparql_after_all_retries(max_regeneration_attempts: int) -> None:
    graph = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.ttl')
    )
    chain = GraphDBQAChain.from_llm(
        Mock(ChatOpenAI), graph=graph, max_regeneration_attempts=max_regeneration_attempts
    )
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_regeneration_select_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_select_chain.run = MagicMock(return_value='```sparql SELECT * {?s ?p ?o} LIMIT 1```')
    chain.sparql_regeneration_select_chain.run = MagicMock(return_value='```sparql SELECT * {?s ?p ?o} LIMIT 1```')

    with pytest.raises(ValueError) as e:
        chain.run('What is Luke Skywalker\'s home planet?')

    assert str(e.value) == 'The generated SPARQL query is invalid.'

    assert chain.sparql_generation_select_chain.run.call_count == 1
    assert chain.sparql_regeneration_select_chain.run.call_count == max_regeneration_attempts
    assert chain.qa_chain.call_count == 0


@pytest.mark.parametrize('max_regeneration_attempts,number_of_invalid_responses', [(1, 0), (2, 0), (2, 1), (10, 6)])
def test_valid_sparql_after_some_retries(max_regeneration_attempts: int, number_of_invalid_responses: int) -> None:
    graph = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.ttl')
    )
    chain = GraphDBQAChain.from_llm(
        Mock(ChatOpenAI), graph=graph, max_regeneration_attempts=max_regeneration_attempts
    )
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_regeneration_select_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    invalid_sparql_query = '```sparql SELECT * {?s ?p ?o} LIMIT 1```'
    valid_sparql_query = 'SELECT * {?s ?p ?o} LIMIT 1'
    chain.sparql_generation_select_chain.run = MagicMock(return_value=invalid_sparql_query)
    chain.sparql_regeneration_select_chain.run = Mock()
    chain.sparql_regeneration_select_chain.run.side_effect = [invalid_sparql_query] * number_of_invalid_responses + [
        valid_sparql_query]

    chain.qa_chain = MagicMock(return_value={
        'text': 'Tatooine',
        'context': [],
        'prompt': 'What is Luke Skywalker\'s home planet?'
    })
    chain.qa_chain.output_key = 'text'

    result = chain.run('What is Luke Skywalker\'s home planet?')

    assert chain.sparql_generation_select_chain.run.call_count == 1
    assert chain.sparql_regeneration_select_chain.run.call_count == number_of_invalid_responses + 1
    assert chain.qa_chain.call_count == 1
    assert result == 'Tatooine'


def query_examples(model_name: str, graph: GraphDBGraph) -> None:
    chain = GraphDBQAChain.from_llm(ChatOpenAI(temperature=0, model_name=model_name), graph=graph)
    try:
        chain.run('What is the average height of the Ewok?')
        chain.run('What is the climate on Tatooine?')
    except ValueError:
        pass


@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])
def test_chain_run_graphdb_query_ontology(model_name: str) -> None:
    graph = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        query_ontology='CONSTRUCT {?s ?p ?o} FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}'
    )
    query_examples(model_name, graph)


@pytest.mark.parametrize("model_name", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])
def test_chain_run_graphdb_local_file(model_name: str) -> None:
    graph = GraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        local_file=str(Path(__file__).parent.parent / "examples/starwars-ontology.ttl")
    )
    query_examples(model_name, graph)
