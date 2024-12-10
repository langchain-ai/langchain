from unittest.mock import MagicMock, Mock

import pytest
from langchain.chains import LLMChain

from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_community.graphs import OntotextGraphDBGraph

"""
cd libs/community/tests/integration_tests/chains/docker-compose-ontotext-graphdb
./start.sh
"""


@pytest.mark.requires("langchain_openai", "rdflib", "SPARQLWrapper")
@pytest.mark.parametrize("max_fix_retries", [-2, -1, 0, 1, 2])
def test_valid_sparql(max_fix_retries: int) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"
    answer = "Tatooine"

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/starwars",
        query_ontology="CONSTRUCT {?s ?p ?o} "
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
        allow_dangerous_requests=True,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": "SELECT * {?s ?p ?o} LIMIT 1",
            "prompt": question,
            "schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = MagicMock()
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock(
        return_value={
            "text": answer,
            "prompt": question,
            "context": [],
        }
    )

    result = chain.invoke({chain.input_key: question})

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == 0
    assert chain.qa_chain.invoke.call_count == 1
    assert result == {chain.output_key: answer, chain.input_key: question}


@pytest.mark.requires("langchain_openai", "rdflib", "SPARQLWrapper")
@pytest.mark.parametrize("max_fix_retries", [-2, -1, 0])
def test_invalid_sparql_non_positive_max_fix_retries(
    max_fix_retries: int,
) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/starwars",
        query_ontology="CONSTRUCT {?s ?p ?o} "
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
        allow_dangerous_requests=True,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": "```sparql SELECT * {?s ?p ?o} LIMIT 1```",
            "prompt": question,
            "schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = MagicMock()
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock()

    from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

    with pytest.raises(QueryBadFormed) as e:
        chain.invoke({chain.input_key: question})

    assert str(e.value) == (
        "QueryBadFormed: A bad request has been sent to the endpoint: "
        "probably the SPARQL query is badly formed. \n\n"
        "Response:\n"
        'b"MALFORMED QUERY: Lexical error at line 1, column 1.  '
        "Encountered: '96' (96),\""
    )

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == 0
    assert chain.qa_chain.invoke.call_count == 0


@pytest.mark.requires("langchain_openai", "rdflib", "SPARQLWrapper")
@pytest.mark.parametrize("max_fix_retries", [1, 2, 3])
def test_valid_sparql_after_first_retry(max_fix_retries: int) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"
    answer = "Tatooine"
    generated_invalid_sparql = "```sparql SELECT * {?s ?p ?o} LIMIT 1```"

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/starwars",
        query_ontology="CONSTRUCT {?s ?p ?o} "
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
        allow_dangerous_requests=True,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "prompt": question,
            "schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = MagicMock(
        return_value={
            "text": "SELECT * {?s ?p ?o} LIMIT 1",
            "error_message": "pyparsing.exceptions.ParseException: "
            "Expected {SelectQuery | ConstructQuery | DescribeQuery | AskQuery}, "
            "found '`'  (at char 0), (line:1, col:1)",
            "generated_sparql": generated_invalid_sparql,
            "schema": "",
        }
    )
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock(
        return_value={
            "text": answer,
            "prompt": question,
            "context": [],
        }
    )

    result = chain.invoke({chain.input_key: question})

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == 1
    assert chain.qa_chain.invoke.call_count == 1
    assert result == {chain.output_key: answer, chain.input_key: question}


@pytest.mark.requires("langchain_openai", "rdflib", "SPARQLWrapper")
@pytest.mark.parametrize("max_fix_retries", [1, 2, 3])
def test_invalid_sparql_after_all_retries(max_fix_retries: int) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"
    generated_invalid_sparql = "```sparql SELECT * {?s ?p ?o} LIMIT 1```"

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/starwars",
        query_ontology="CONSTRUCT {?s ?p ?o} "
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
        allow_dangerous_requests=True,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "prompt": question,
            "schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "error_message": "pyparsing.exceptions.ParseException: "
            "Expected {SelectQuery | ConstructQuery | DescribeQuery | AskQuery}, "
            "found '`'  (at char 0), (line:1, col:1)",
            "generated_sparql": generated_invalid_sparql,
            "schema": "",
        }
    )
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock()

    from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

    with pytest.raises(QueryBadFormed) as e:
        chain.invoke({chain.input_key: question})

    assert str(e.value) == (
        "QueryBadFormed: A bad request has been sent to the endpoint: "
        "probably the SPARQL query is badly formed. \n\n"
        "Response:\n"
        'b"MALFORMED QUERY: Lexical error at line 1, column 1.  '
        "Encountered: '96' (96),\""
    )

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == max_fix_retries
    assert chain.qa_chain.invoke.call_count == 0


@pytest.mark.requires("langchain_openai", "rdflib", "SPARQLWrapper")
@pytest.mark.parametrize(
    "max_fix_retries,number_of_invalid_responses",
    [(1, 0), (2, 0), (2, 1), (10, 6)],
)
def test_valid_sparql_after_some_retries(
    max_fix_retries: int, number_of_invalid_responses: int
) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"
    answer = "Tatooine"
    generated_invalid_sparql = "```sparql SELECT * {?s ?p ?o} LIMIT 1```"
    generated_valid_sparql_query = "SELECT * {?s ?p ?o} LIMIT 1"

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/starwars",
        query_ontology="CONSTRUCT {?s ?p ?o} "
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
        allow_dangerous_requests=True,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "prompt": question,
            "schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = Mock()
    chain.sparql_fix_chain.invoke.side_effect = [
        {
            "text": generated_invalid_sparql,
            "error_message": "pyparsing.exceptions.ParseException: "
            "Expected {SelectQuery | ConstructQuery | DescribeQuery | AskQuery}, "
            "found '`'  (at char 0), (line:1, col:1)",
            "generated_sparql": generated_invalid_sparql,
            "schema": "",
        }
    ] * number_of_invalid_responses + [
        {
            "text": generated_valid_sparql_query,
            "error_message": "pyparsing.exceptions.ParseException: "
            "Expected {SelectQuery | ConstructQuery | DescribeQuery | AskQuery}, "
            "found '`'  (at char 0), (line:1, col:1)",
            "generated_sparql": generated_invalid_sparql,
            "schema": "",
        }
    ]
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock(
        return_value={
            "text": answer,
            "prompt": question,
            "context": [],
        }
    )

    result = chain.invoke({chain.input_key: question})

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == number_of_invalid_responses + 1
    assert chain.qa_chain.invoke.call_count == 1
    assert result == {chain.output_key: answer, chain.input_key: question}


@pytest.mark.requires("langchain_openai", "rdflib", "SPARQLWrapper")
@pytest.mark.parametrize(
    "model_name,question",
    [
        ("gpt-3.5-turbo-1106", "What is the average height of the Wookiees?"),
        ("gpt-3.5-turbo-1106", "What is the climate on Tatooine?"),
        ("gpt-3.5-turbo-1106", "What is Luke Skywalker's home planet?"),
        ("gpt-4-1106-preview", "What is the average height of the Wookiees?"),
        ("gpt-4-1106-preview", "What is the climate on Tatooine?"),
        ("gpt-4-1106-preview", "What is Luke Skywalker's home planet?"),
    ],
)
def test_chain(model_name: str, question: str) -> None:
    from langchain_openai import ChatOpenAI

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/starwars",
        query_ontology="CONSTRUCT {?s ?p ?o} "
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        ChatOpenAI(temperature=0, model_name=model_name),  # type: ignore[call-arg]
        graph=graph,
        verbose=True,  # type: ignore[call-arg]
        allow_dangerous_requests=True,
    )
    try:
        chain.invoke({chain.input_key: question})
    except ValueError:
        pass
