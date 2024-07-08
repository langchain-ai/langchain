"""Test RDF/ SPARQL Graph Database Chain."""

import pathlib
import re
from unittest.mock import MagicMock, Mock

from langchain.chains import LLMChain

from langchain_community.chains.graph_qa.sparql import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph

"""
cd libs/langchain/tests/integration_tests/chains/docker-compose-ontotext-graphdb
./start.sh
"""


def test_connect_file_rdf() -> None:
    """
    Test loading online resource.
    """
    berners_lee_card = "http://www.w3.org/People/Berners-Lee/card"

    graph = RdfGraph(
        source_file=berners_lee_card,
        standard="rdf",
    )

    query = """SELECT ?s ?p ?o\n""" """WHERE { ?s ?p ?o }"""

    output = graph.query(query)
    assert len(output) == 86


def test_sparql_select() -> None:
    """
    Test for generating and executing simple SPARQL SELECT query.
    """
    from langchain_openai import ChatOpenAI

    berners_lee_card = "http://www.w3.org/People/Berners-Lee/card"

    graph = RdfGraph(
        source_file=berners_lee_card,
        standard="rdf",
    )

    question = "What is Tim Berners-Lee's work homepage?"
    answer = "Tim Berners-Lee's work homepage is http://www.w3.org/People/Berners-Lee/."

    chain = GraphSparqlQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
    )
    chain.sparql_intent_chain = Mock(LLMChain)
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_generation_update_chain = Mock(LLMChain)

    chain.sparql_intent_chain.run = Mock(return_value="SELECT")
    chain.sparql_generation_select_chain.run = Mock(
        return_value="""PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        SELECT ?workHomepage
                        WHERE {
                            ?person rdfs:label "Tim Berners-Lee" .
                            ?person foaf:workplaceHomepage ?workHomepage .
                        }"""
    )
    chain.qa_chain = MagicMock(
        return_value={
            "text": answer,
            "prompt": question,
            "context": [],
        }
    )
    chain.qa_chain.output_key = "text"

    output = chain.invoke({chain.input_key: question})[chain.output_key]
    assert output == answer

    assert chain.sparql_intent_chain.run.call_count == 1
    assert chain.sparql_generation_select_chain.run.call_count == 1
    assert chain.sparql_generation_update_chain.run.call_count == 0
    assert chain.qa_chain.call_count == 1


def test_sparql_insert(tmp_path: pathlib.Path) -> None:
    """
    Test for generating and executing simple SPARQL INSERT query.
    """
    from langchain_openai import ChatOpenAI

    berners_lee_card = "http://www.w3.org/People/Berners-Lee/card"
    local_copy = tmp_path / "test.ttl"

    graph = RdfGraph(
        source_file=berners_lee_card,
        standard="rdf",
        local_copy=str(local_copy),
    )

    query = (
        "Save that the person with the name 'Timothy Berners-Lee' "
        "has a work homepage at 'http://www.w3.org/foo/bar/'"
    )

    chain = GraphSparqlQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
    )
    chain.sparql_intent_chain = Mock(LLMChain)
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_generation_update_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_intent_chain.run = Mock(return_value="UPDATE")
    chain.sparql_generation_update_chain.run = Mock(
        return_value="""PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                        INSERT {
                            ?p foaf:workplaceHomepage <http://www.w3.org/foo/bar/> .
                        }
                        WHERE {
                            ?p foaf:name "Timothy Berners-Lee" .
                        }"""
    )

    output = chain.invoke({chain.input_key: query})[chain.output_key]
    assert output == "Successfully inserted triples into the graph."

    assert chain.sparql_intent_chain.run.call_count == 1
    assert chain.sparql_generation_select_chain.run.call_count == 0
    assert chain.sparql_generation_update_chain.run.call_count == 1
    assert chain.qa_chain.call_count == 0

    query = (
        """PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n"""
        """SELECT ?hp\n"""
        """WHERE {\n"""
        """    ?person foaf:name "Timothy Berners-Lee" . \n"""
        """    ?person foaf:workplaceHomepage ?hp .\n"""
        """}"""
    )
    output = graph.query(query)
    assert len(output) == 2


def test_sparql_select_return_query() -> None:
    """
    Test for generating and executing simple SPARQL SELECT query
    and returning the generated SPARQL query.
    """
    from langchain_openai import ChatOpenAI

    berners_lee_card = "http://www.w3.org/People/Berners-Lee/card"

    graph = RdfGraph(
        source_file=berners_lee_card,
        standard="rdf",
    )

    question = "What is Tim Berners-Lee's work homepage?"
    answer = "Tim Berners-Lee's work homepage is http://www.w3.org/People/Berners-Lee/."

    chain = GraphSparqlQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        return_sparql_query=True,
    )
    chain.sparql_intent_chain = Mock(LLMChain)
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_generation_update_chain = Mock(LLMChain)

    chain.sparql_intent_chain.run = Mock(return_value="SELECT")
    chain.sparql_generation_select_chain.run = Mock(
        return_value="""PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        SELECT ?workHomepage
                        WHERE {
                            ?person rdfs:label "Tim Berners-Lee" .
                            ?person foaf:workplaceHomepage ?workHomepage .
                        }"""
    )
    chain.qa_chain = MagicMock(
        return_value={
            "text": answer,
            "prompt": question,
            "context": [],
        }
    )
    chain.qa_chain.output_key = "text"

    output = chain.invoke({chain.input_key: question})
    assert output[chain.output_key] == answer
    assert "sparql_query" in output

    assert chain.sparql_intent_chain.run.call_count == 1
    assert chain.sparql_generation_select_chain.run.call_count == 1
    assert chain.sparql_generation_update_chain.run.call_count == 0
    assert chain.qa_chain.call_count == 1


def test_loading_schema_from_ontotext_graphdb() -> None:
    graph = RdfGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        graph_kwargs={"bind_namespaces": "none"},
    )
    schema = graph.get_schema
    prefix = (
        "In the following, each IRI is followed by the local name and "
        "optionally its description in parentheses. \n"
        "The RDF graph supports the following node types:"
    )
    assert schema.startswith(prefix)

    infix = "The RDF graph supports the following relationships:"
    assert infix in schema

    classes = schema[len(prefix) : schema.index(infix)]
    assert len(re.findall("<[^>]+> \\([^)]+\\)", classes)) == 5

    relationships = schema[schema.index(infix) + len(infix) :]
    assert len(re.findall("<[^>]+> \\([^)]+\\)", relationships)) == 58


def test_graph_qa_chain_with_ontotext_graphdb() -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Tim Berners-Lee's work homepage?"
    answer = "Tim Berners-Lee's work homepage is http://www.w3.org/People/Berners-Lee/."

    graph = RdfGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        graph_kwargs={"bind_namespaces": "none"},
    )

    chain = GraphSparqlQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
    )
    chain.sparql_intent_chain = Mock(LLMChain)
    chain.sparql_generation_select_chain = Mock(LLMChain)
    chain.sparql_generation_update_chain = Mock(LLMChain)

    chain.sparql_intent_chain.run = Mock(return_value="SELECT")
    chain.sparql_generation_select_chain.run = Mock(
        return_value="""PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        SELECT ?workHomepage
                        WHERE {
                            ?person rdfs:label "Tim Berners-Lee" .
                            ?person foaf:workplaceHomepage ?workHomepage .
                        }"""
    )
    chain.qa_chain = MagicMock(
        return_value={
            "text": answer,
            "prompt": question,
            "context": [],
        }
    )
    chain.qa_chain.output_key = "text"

    output = chain.invoke({chain.input_key: question})[chain.output_key]
    assert output == answer

    assert chain.sparql_intent_chain.run.call_count == 1
    assert chain.sparql_generation_select_chain.run.call_count == 1
    assert chain.sparql_generation_update_chain.run.call_count == 0
    assert chain.qa_chain.call_count == 1
