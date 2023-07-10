import pytest

from langchain import SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_sql_chain_without_memory() -> None:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    assert db_chain.run("hello") == "Final Answer: baz"


def test_sql_chain_with_valid_memory() -> None:
    valid_prompt_with_history = """
    Only use the following tables:
    {table_info}
    Question: {input}
    
    Given an input question, first create a syntactically correct {dialect} query to run.
    Always limit your query to at most {top_k} results.

    Relevant pieces of previous conversation:
    {history}

    (You do not need to use these pieces of information if not relevant)
    """
    prompt = PromptTemplate(
        input_variables=["input", "table_info", "dialect", "top_k", "history"],
        template=valid_prompt_with_history,
    )
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    memory = ConversationBufferMemory(memory_key="history", input_key="input")
    db_chain = SQLDatabaseChain.from_llm(
        llm, db, memory=memory, prompt=prompt, verbose=True
    )
    assert db_chain.run("hello") == "Final Answer: baz"


def test_sql_chain_invalid_memory() -> None:
    invalid_prompt_without_history = """
    Only use the following tables:
    {table_info}
    Question: {input}
    
    Given an input question, first create a syntactically correct {dialect} query to run.
    Always limit your query to at most {top_k} results.
    """
    prompt = PromptTemplate(
        input_variables=["input", "table_info", "dialect", "top_k"],
        template=invalid_prompt_without_history,
    )
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    memory = ConversationBufferMemory(memory_key="history", input_key="input")
    with pytest.raises(ValueError):
        db_chain = SQLDatabaseChain.from_llm(
            llm, db, memory=memory, prompt=prompt, verbose=True
        )
