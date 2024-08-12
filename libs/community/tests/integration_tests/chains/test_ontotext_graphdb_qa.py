from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from langchain.chains import LLMChain
from langchain_core.prompts.prompt import PromptTemplate

from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_community.graphs import OntotextGraphDBGraph

"""
cd libs/community/tests/integration_tests/chains/docker-compose-ontotext-graphdb
./start.sh
"""


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
@pytest.mark.parametrize("max_fix_retries", [-2, -1, 0, 1, 2])
def test_valid_sparql(max_fix_retries: int) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"
    answer = "Tatooine"
    generated_sparql = "SELECT * {?s ?p ?o} LIMIT 1"

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_sparql,
            "question": question,
            "ontology_schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = MagicMock()
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock(
        return_value={
            "text": answer,
            "question": question,
            "ontology_schema": "",
            "context": [],
        }
    )

    inputs = {"question": question, "ontology_schema": ""}
    result = chain.invoke(inputs)

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == 0
    assert chain.qa_chain.invoke.call_count == 1
    inputs.update(
        {
            chain.output_key_generated_sparql: generated_sparql,
            chain.output_key_answer: answer,
        }
    )
    assert result == inputs


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
@pytest.mark.parametrize("max_fix_retries", [-2, -1, 0])
def test_invalid_sparql_non_positive_max_fix_retries(
    max_fix_retries: int,
) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": "```sparql SELECT * {?s ?p ?o} LIMIT 1```",
            "question": question,
            "ontology_schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = MagicMock()
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock()

    from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

    inputs = {"question": question, "ontology_schema": ""}
    with pytest.raises(QueryBadFormed) as e:
        chain.invoke(inputs)

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


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
@pytest.mark.parametrize("max_fix_retries", [1, 2, 3])
def test_valid_sparql_after_first_retry(max_fix_retries: int) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"
    answer = "Tatooine"
    generated_invalid_sparql = "```sparql SELECT * {?s ?p ?o} LIMIT 1```"
    generated_valid_sparql = "SELECT * {?s ?p ?o} LIMIT 1"

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = MagicMock(
        return_value={
            "text": generated_valid_sparql,
            "error_message": (
                "QueryBadFormed: A bad request has been sent to the endpoint: "
                "probably the SPARQL query is badly formed. \n\n"
                "Response:"
                'b"MALFORMED QUERY: Lexical error at line 1, column 1.  '
                "Encountered: '96' (96),\""
            ),
            "generated_sparql": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
        }
    )
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock(
        return_value={
            "text": answer,
            "question": question,
            "ontology_schema": "",
            "context": [],
        }
    )

    inputs = {"question": question, "ontology_schema": ""}
    result = chain.invoke(inputs)

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == 1
    assert chain.qa_chain.invoke.call_count == 1
    inputs.update(
        {
            chain.output_key_generated_sparql: generated_valid_sparql,
            chain.output_key_answer: answer,
        }
    )
    assert result == inputs


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
@pytest.mark.parametrize("max_fix_retries", [1, 2, 3])
def test_invalid_sparql_after_all_retries(max_fix_retries: int) -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"
    generated_invalid_sparql = "```sparql SELECT * {?s ?p ?o} LIMIT 1```"

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "error_message": (
                "QueryBadFormed: A bad request has been sent to the endpoint: "
                "probably the SPARQL query is badly formed. \n\n"
                "Response:"
                'b"MALFORMED QUERY: Lexical error at line 1, column 1.  '
                "Encountered: '96' (96),\""
            ),
            "generated_sparql": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
        }
    )
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock()

    from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

    inputs = {"question": question, "ontology_schema": ""}
    with pytest.raises(QueryBadFormed) as e:
        chain.invoke(inputs)

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


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
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
    generated_valid_sparql = "SELECT * {?s ?p ?o} LIMIT 1"

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        max_fix_retries=max_fix_retries,
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = Mock()
    chain.sparql_fix_chain.invoke.side_effect = [
        {
            "text": generated_invalid_sparql,
            "error_message": (
                "QueryBadFormed: A bad request has been sent to the endpoint: "
                "probably the SPARQL query is badly formed. \n\n"
                "Response:"
                'b"MALFORMED QUERY: Lexical error at line 1, column 1.  '
                "Encountered: '96' (96),\""
            ),
            "generated_sparql": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
        }
    ] * number_of_invalid_responses + [
        {
            "text": generated_valid_sparql,
            "error_message": (
                "QueryBadFormed: A bad request has been sent to the endpoint: "
                "probably the SPARQL query is badly formed. \n\n"
                "Response:"
                'b"MALFORMED QUERY: Lexical error at line 1, column 1.  '
                "Encountered: '96' (96),\""
            ),
            "generated_sparql": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
        }
    ]
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock(
        return_value={
            "text": answer,
            "question": question,
            "ontology_schema": "",
            "context": [],
        }
    )

    inputs = {"question": question, "ontology_schema": ""}
    result = chain.invoke(inputs)

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == number_of_invalid_responses + 1
    assert chain.qa_chain.invoke.call_count == 1
    inputs.update(
        {
            chain.output_key_generated_sparql: generated_valid_sparql,
            chain.output_key_answer: answer,
        }
    )
    assert result == inputs


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
def test_custom_sparql_generation_prompt_variables() -> None:
    from langchain_openai import ChatOpenAI

    question = "What is Luke Skywalker's home planet?"
    answer = "Tatooine"
    generated_valid_sparql = "SELECT * {?s ?p ?o} LIMIT 1"

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )
    chain = OntotextGraphDBQAChain.from_llm(
        Mock(ChatOpenAI),
        graph=graph,
        sparql_generation_prompt=PromptTemplate(
            input_variables=["x"],
            template="""{x}""",
        ),
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_fix_chain = Mock(LLMChain)
    chain.qa_chain = Mock(LLMChain)

    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_valid_sparql,
            "question": question,
            "ontology_schema": "",
            "x": "",
        }
    )
    chain.sparql_fix_chain.output_key = "text"
    chain.sparql_fix_chain.invoke = Mock()
    chain.qa_chain.output_key = "text"
    chain.qa_chain.invoke = MagicMock(
        return_value={
            "text": answer,
            "question": question,
            "ontology_schema": "",
            "x": "",
            "context": [],
        }
    )

    inputs = {"w": question}
    with pytest.raises(ValueError) as e:
        chain.invoke(inputs)

    assert str(e.value) == "Missing some input keys: {'x'}"

    assert chain.sparql_generation_chain.invoke.call_count == 0
    assert chain.sparql_fix_chain.invoke.call_count == 0
    assert chain.qa_chain.invoke.call_count == 0

    inputs = {"x": question}
    result = chain.invoke(inputs)

    assert chain.sparql_generation_chain.invoke.call_count == 1
    assert chain.sparql_fix_chain.invoke.call_count == 0
    assert chain.qa_chain.invoke.call_count == 1
    inputs.update(
        {
            chain.output_key_generated_sparql: generated_valid_sparql,
            chain.output_key_answer: answer,
        }
    )
    assert result == inputs


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
def test_chain() -> None:
    from langchain_openai import ChatOpenAI

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )

    chain = OntotextGraphDBQAChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-2024-05-13",
            seed=123,
        ),
        graph=graph,
    )

    schema_file_path = Path(__file__).parent.parent / "examples" / "ontology-schema.ttl"
    inputs = {
        "question": "What is the average height of the Wookiees?",
        "ontology_schema": schema_file_path.read_text(encoding="utf-8"),
    }
    chain.invoke(inputs)


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
def test_chain_custom_generation_prompt() -> None:
    from langchain_openai import ChatOpenAI

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )

    template = """
Write a SPARQL SELECT query to answer the user question 
delimited by triple backticks:\n```{question}```\n
The question mentions the following concepts in JSON format 
delimited by triple backticks\n```{named_entities}```\n
The ontology schema delimited by triple backticks in 
Turtle format is:\n```{ontology_schema}```\n
Use only the classes and properties provided in the schema 
to construct the SPARQL query.
Do not use any classes or properties 
that are not explicitly provided in the SPARQL query.
Include all necessary prefixes.
Do not include any explanations or apologies in your responses.
Do not wrap the query in backticks.
Do not include any text except the SPARQL query generated.
"""
    chain = OntotextGraphDBQAChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-2024-05-13",
            seed=123,
        ),
        graph=graph,
        sparql_generation_prompt=PromptTemplate(
            input_variables=["question", "named_entities", "ontology_schema"],
            template=template,
        ),
    )
    schema_file_path = Path(__file__).parent.parent / "examples" / "ontology-schema.ttl"
    inputs = {
        "question": "What is Luke Skywalker's home planet?",
        "ontology_schema": schema_file_path.read_text(encoding="utf-8"),
        "named_entities": [
            {
                "class": "https://swapi.co/vocabulary/Human",
                "inst": "https://swapi.co/resource/human/1",
            },
        ],
    }
    chain.invoke(inputs)


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
def test_custom_sparql_fix_prompt() -> None:
    from langchain_openai import ChatOpenAI

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )

    chain = OntotextGraphDBQAChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-2024-05-13",
            seed=123,
        ),
        graph=graph,
        sparql_fix_prompt=PromptTemplate(
            input_variables=["x", "error_message", "generated_sparql"],
            template="{x} {error_message} {generated_sparql}",
        ),
    )

    question = "What is Luke Skywalker's home planet?"
    generated_invalid_sparql = "```sparql SELECT * {?s ?p ?o} LIMIT 1```"

    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
        }
    )

    inputs = {
        "question": question,
        "ontology_schema": "",
    }
    with pytest.raises(ValueError) as e:
        chain.invoke(inputs)
    assert str(e.value) == "Missing some input keys: {'x'}"

    assert chain.sparql_generation_chain.invoke.call_count == 1

    template = """{x}
The following SPARQL query delimited by triple backticks
```
{generated_sparql}
```
is not valid.
The error delimited by triple backticks is
```
{error_message}
```
Give me a correct version of the SPARQL query.
Do not change the logic of the query.
Do not include any explanations or apologies in your responses.
Do not wrap the query in backticks.
Do not include any text except the SPARQL query generated.
The ontology schema delimited by triple backticks in Turtle format is:
```
{ontology_schema}
```
"""
    chain = OntotextGraphDBQAChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-2024-05-13",
            seed=123,
        ),
        graph=graph,
        sparql_fix_prompt=PromptTemplate(
            input_variables=[
                "x",
                "generated_sparql",
                "error_message",
                "ontology_schema",
            ],
            template=template,
        ),
    )
    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_invalid_sparql,
            "question": question,
            "ontology_schema": "",
            "x": "",
        }
    )

    schema_file_path = Path(__file__).parent.parent / "examples" / "ontology-schema.ttl"
    inputs = {
        "question": question,
        "ontology_schema": schema_file_path.read_text(encoding="utf-8"),
        "x": "",
    }
    chain.invoke(inputs)

    assert chain.sparql_generation_chain.invoke.call_count == 1


@pytest.mark.requires("langchain_openai", "SPARQLWrapper")
def test_custom_qa_prompt() -> None:
    from langchain_openai import ChatOpenAI

    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/starwars",
    )

    chain = OntotextGraphDBQAChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-2024-05-13",
            seed=123,
        ),
        graph=graph,
        qa_prompt=PromptTemplate(
            input_variables=["x", "context"],
            template="{x} {context}",
        ),
    )
    question = "What is Luke Skywalker's home planet?"

    chain.sparql_generation_chain = Mock(LLMChain)
    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": "SELECT * {?s ?p ?o} LIMIT 1",
            "question": question,
            "ontology_schema": "",
        }
    )

    inputs = {
        "question": question,
        "ontology_schema": "",
    }
    with pytest.raises(ValueError) as e:
        chain.invoke(inputs)
    assert str(e.value) == "Missing some input keys: {'x'}"

    assert chain.sparql_generation_chain.invoke.call_count == 1

    template = """{x} The data ```{context}``` is the answer 
to the question \"\"\"{question}\"\"\". 
Generate a human readable answer to the question using the provided data. 
"""
    chain = OntotextGraphDBQAChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-2024-05-13",
            seed=123,
        ),
        graph=graph,
        sparql_fix_prompt=PromptTemplate(
            input_variables=["x", "context", "question"],
            template=template,
        ),
    )

    chain.sparql_generation_chain = Mock(LLMChain)
    generated_valid_sparql = (
        "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        "PREFIX voc: <https://swapi.co/vocabulary/> "
        "SELECT ?homePlanet "
        "WHERE { "
        "  <https://swapi.co/resource/human/1> voc:homeworld / rdfs:label ?homePlanet"
        "}"
    )
    chain.sparql_generation_chain.output_key = "text"
    chain.sparql_generation_chain.invoke = MagicMock(
        return_value={
            "text": generated_valid_sparql,
            "question": question,
            "ontology_schema": "",
            "x": "",
        }
    )

    schema_file_path = Path(__file__).parent.parent / "examples" / "ontology-schema.ttl"
    inputs = {
        "question": question,
        "ontology_schema": schema_file_path.read_text(encoding="utf-8"),
        "x": "",
    }
    chain.invoke(inputs)

    assert chain.sparql_generation_chain.invoke.call_count == 1
