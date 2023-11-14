from typing import Any, Dict, List

import pandas as pd

from langchain.chains.graph_qa.cypher import (
    GraphCypherQAChain,
    construct_schema,
    extract_cypher,
)
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.graphs.graph_document import GraphDocument
from langchain.graphs.graph_store import GraphStore
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts import PromptTemplate
from tests.unit_tests.llms.fake_llm import FakeLLM


class FakeGraphStore(GraphStore):
    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph database"""
        return ""

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph database"""
        return {}

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query the graph."""
        return []

    def refresh_schema(self) -> None:
        """Refreshes the graph schema information."""
        pass

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input as uses it to construct a graph."""
        pass


def test_graph_cypher_qa_chain_prompt_selection_1() -> None:
    # Pass prompts directly. No kwargs is specified.
    qa_prompt_template = "QA Prompt"
    cypher_prompt_template = "Cypher Prompt"
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=[])
    cypher_prompt = PromptTemplate(template=cypher_prompt_template, input_variables=[])
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        qa_prompt=qa_prompt,
        cypher_prompt=cypher_prompt,
    )
    assert chain.qa_chain.prompt == qa_prompt
    assert chain.cypher_generation_chain.prompt == cypher_prompt


def test_graph_cypher_qa_chain_prompt_selection_2() -> None:
    # Default case. Pass nothing
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
    )
    assert chain.qa_chain.prompt == CYPHER_QA_PROMPT
    assert chain.cypher_generation_chain.prompt == CYPHER_GENERATION_PROMPT


def test_graph_cypher_qa_chain_prompt_selection_3() -> None:
    # Pass non-prompt args only to sub-chains via kwargs
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        cypher_llm_kwargs={"memory": readonlymemory},
        qa_llm_kwargs={"memory": readonlymemory},
    )
    assert chain.qa_chain.prompt == CYPHER_QA_PROMPT
    assert chain.cypher_generation_chain.prompt == CYPHER_GENERATION_PROMPT


def test_graph_cypher_qa_chain_prompt_selection_4() -> None:
    # Pass prompt, non-prompt args to subchains via kwargs
    qa_prompt_template = "QA Prompt"
    cypher_prompt_template = "Cypher Prompt"
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=[])
    cypher_prompt = PromptTemplate(template=cypher_prompt_template, input_variables=[])
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        cypher_llm_kwargs={"prompt": cypher_prompt, "memory": readonlymemory},
        qa_llm_kwargs={"prompt": qa_prompt, "memory": readonlymemory},
    )
    assert chain.qa_chain.prompt == qa_prompt
    assert chain.cypher_generation_chain.prompt == cypher_prompt


def test_graph_cypher_qa_chain_prompt_selection_5() -> None:
    # Can't pass both prompt and kwargs at the same time
    qa_prompt_template = "QA Prompt"
    cypher_prompt_template = "Cypher Prompt"
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=[])
    cypher_prompt = PromptTemplate(template=cypher_prompt_template, input_variables=[])
    try:
        GraphCypherQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
            verbose=True,
            return_intermediate_steps=False,
            qa_prompt=qa_prompt,
            cypher_prompt=cypher_prompt,
            cypher_llm_kwargs={"memory": readonlymemory},
            qa_llm_kwargs={"memory": readonlymemory},
        )
        assert False
    except ValueError:
        assert True


def test_graph_cypher_qa_chain() -> None:
    template = """You are a nice chatbot having a conversation with a human.

    Schema:
    {schema}

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:"""

    prompt = PromptTemplate(
        input_variables=["schema", "question", "chat_history"], template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    prompt1 = (
        "You are a nice chatbot having a conversation with a human.\n\n    "
        "Schema:\n    Node properties are the following: \n {}\nRelationships "
        "properties are the following: \n {}\nRelationships are: \n[]\n\n    "
        "Previous conversation:\n    \n\n    New human question: "
        "Test question\n    Response:"
    )

    prompt2 = (
        "You are a nice chatbot having a conversation with a human.\n\n    "
        "Schema:\n    Node properties are the following: \n {}\nRelationships "
        "properties are the following: \n {}\nRelationships are: \n[]\n\n    "
        "Previous conversation:\n    Human: Test question\nAI: foo\n\n    "
        "New human question: Test new question\n    Response:"
    )

    llm = FakeLLM(queries={prompt1: "answer1", prompt2: "answer2"})
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        cypher_llm_kwargs={"prompt": prompt, "memory": readonlymemory},
        memory=memory,
    )
    chain.run("Test question")
    chain.run("Test new question")
    # If we get here without a key error, that means memory
    # was used properly to create prompts.
    assert True


def test_no_backticks() -> None:
    """Test if there are no backticks, so the original text should be returned."""
    query = "MATCH (n) RETURN n"
    output = extract_cypher(query)
    assert output == query


def test_backticks() -> None:
    """Test if there are backticks. Query from within backticks should be returned."""
    query = "You can use the following query: ```MATCH (n) RETURN n```"
    output = extract_cypher(query)
    assert output == "MATCH (n) RETURN n"


def test_exclude_types() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    exclude_types = ["Person", "DIRECTED"]
    output = construct_schema(structured_schema, [], exclude_types)
    expected_schema = (
        "Node properties are the following: \n"
        " {'Movie': [{'property': 'title', 'type': 'STRING'}], "
        "'Actor': [{'property': 'name', 'type': 'STRING'}]}\n"
        "Relationships properties are the following: \n"
        " {}\nRelationships are: \n"
        "['(:Actor)-[:ACTED_IN]->(:Movie)']"
    )
    assert output == expected_schema


def test_include_types() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor", "ACTED_IN"]
    output = construct_schema(structured_schema, include_types, [])
    expected_schema = (
        "Node properties are the following: \n"
        " {'Movie': [{'property': 'title', 'type': 'STRING'}], "
        "'Actor': [{'property': 'name', 'type': 'STRING'}]}\n"
        "Relationships properties are the following: \n"
        " {}\nRelationships are: \n"
        "['(:Actor)-[:ACTED_IN]->(:Movie)']"
    )
    assert output == expected_schema


def test_include_types2() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor"]
    output = construct_schema(structured_schema, include_types, [])
    expected_schema = (
        "Node properties are the following: \n"
        " {'Movie': [{'property': 'title', 'type': 'STRING'}], "
        "'Actor': [{'property': 'name', 'type': 'STRING'}]}\n"
        "Relationships properties are the following: \n"
        " {}\nRelationships are: \n"
        "[]"
    )
    assert output == expected_schema


def test_include_types3() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor", "ACTED_IN"]
    output = construct_schema(structured_schema, include_types, [])
    expected_schema = (
        "Node properties are the following: \n"
        " {'Movie': [{'property': 'title', 'type': 'STRING'}], "
        "'Actor': [{'property': 'name', 'type': 'STRING'}]}\n"
        "Relationships properties are the following: \n"
        " {}\nRelationships are: \n"
        "['(:Actor)-[:ACTED_IN]->(:Movie)']"
    )
    assert output == expected_schema


def test_validating_cypher_statements() -> None:
    cypher_file = "tests/unit_tests/data/cypher_corrector.csv"
    examples = pd.read_csv(cypher_file)
    examples.fillna("", inplace=True)
    for _, row in examples.iterrows():
        schema = load_schemas(row["schema"])
        corrector = CypherQueryCorrector(schema)
        assert corrector(row["statement"]) == row["correct_query"]


def load_schemas(str_schemas: str) -> List[Schema]:
    """
    Args:
        str_schemas: string of schemas
    """
    values = str_schemas.replace("(", "").replace(")", "").split(",")
    schemas = []
    for i in range(len(values) // 3):
        schemas.append(
            Schema(
                values[i * 3].strip(),
                values[i * 3 + 1].strip(),
                values[i * 3 + 2].strip(),
            )
        )
    return schemas
