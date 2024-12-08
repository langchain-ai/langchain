import pathlib
from typing import Any, Dict, List

import pandas as pd
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_core.prompts import PromptTemplate

from langchain_community.chains.graph_qa.cypher import (
    GraphCypherQAChain,
    construct_schema,
    extract_cypher,
)
from langchain_community.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
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
        allow_dangerous_requests=True,
    )
    assert chain.qa_chain.prompt == qa_prompt  # type: ignore[union-attr]
    assert chain.cypher_generation_chain.prompt == cypher_prompt


def test_graph_cypher_qa_chain_prompt_selection_2() -> None:
    # Default case. Pass nothing
    chain = GraphCypherQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        verbose=True,
        return_intermediate_steps=False,
        allow_dangerous_requests=True,
    )
    assert chain.qa_chain.prompt == CYPHER_QA_PROMPT  # type: ignore[union-attr]
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
        allow_dangerous_requests=True,
    )
    assert chain.qa_chain.prompt == CYPHER_QA_PROMPT  # type: ignore[union-attr]
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
        allow_dangerous_requests=True,
    )
    assert chain.qa_chain.prompt == qa_prompt  # type: ignore[union-attr]
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
            allow_dangerous_requests=True,
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
        "Schema:\n    Node properties are the following:\n\nRelationship "
        "properties are the following:\n\nThe relationships are the "
        "following:\n\n\n    "
        "Previous conversation:\n    \n\n    New human question: "
        "Test question\n    Response:"
    )

    prompt2 = (
        "You are a nice chatbot having a conversation with a human.\n\n    "
        "Schema:\n    Node properties are the following:\n\nRelationship "
        "properties are the following:\n\nThe relationships are the "
        "following:\n\n\n    "
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
        allow_dangerous_requests=True,
    )
    chain.run("Test question")
    chain.run("Test new question")
    # If we get here without a key error, that means memory
    # was used properly to create prompts.
    assert True


def test_extract_cypher_with_no_backticks() -> None:
    """Test if there are no backticks, so the original text should be returned."""
    query = "MATCH (n) RETURN n"
    output = extract_cypher(query)
    assert output == query


def test_extract_cypher_with_backticks() -> None:
    """Test if there are backticks. Query from within backticks should be returned."""
    query = "You can use the following query: ```MATCH (n) RETURN n```"
    output = extract_cypher(query)
    assert output == "MATCH (n) RETURN n"


def test_extract_cypher_with_node_name_with_backticks() -> None:
    """Test if there are backticks in the node name.
    The original text should be returned."""
    query = "MATCH (n: `Node Name`) RETURN n"
    output = extract_cypher(query)
    assert output == "MATCH (n: `Node Name`) RETURN n"


def test_extract_cypher_with_node_name_with_one_space() -> None:
    """Test if there is one space in the node name.
    Node name should be wrapped in backticks."""
    query = "MATCH (n: Node Name) RETURN n"
    output = extract_cypher(query)
    assert output == "MATCH (n: `Node Name`) RETURN n"


def test_extract_cypher_with_node_name_with_multi_spaces() -> None:
    """Test if there are multiple spaces in the node name.
    Node name should be wrapped in backticks."""
    query = "MATCH (n: Node Name With Multiple Spaces) RETURN n"
    output = extract_cypher(query)
    assert output == "MATCH (n: `Node Name With Multiple Spaces`) RETURN n"


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
        "Node properties are the following:\n"
        "Movie {title: STRING},Actor {name: STRING}\n"
        "Relationship properties are the following:\n\n"
        "The relationships are the following:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
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
        "Node properties are the following:\n"
        "Movie {title: STRING},Actor {name: STRING}\n"
        "Relationship properties are the following:\n\n"
        "The relationships are the following:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
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
        "Node properties are the following:\n"
        "Movie {title: STRING},Actor {name: STRING}\n"
        "Relationship properties are the following:\n\n"
        "The relationships are the following:\n"
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
        "Node properties are the following:\n"
        "Movie {title: STRING},Actor {name: STRING}\n"
        "Relationship properties are the following:\n\n"
        "The relationships are the following:\n"
        "(:Actor)-[:ACTED_IN]->(:Movie)"
    )
    assert output == expected_schema


HERE = pathlib.Path(__file__).parent

UNIT_TESTS_ROOT = HERE.parent


def test_validating_cypher_statements() -> None:
    cypher_file = str(UNIT_TESTS_ROOT / "data/cypher_corrector.csv")
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
