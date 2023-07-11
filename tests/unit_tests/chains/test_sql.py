import pytest

from langchain import SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_sql_chain_without_memory() -> None:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    assert db_chain.run("hello") == "Final Answer: baz"


def test_sql_chain_sequential_without_memory() -> None:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True)
    assert db_chain.run("hello") == "Final Answer: baz"


def test_sql_chain_with_memory() -> None:
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


def test_sql_chain_sequential_with_memory() -> None:
    valid_query_prompt_str = """
    Only use the following tables:
    {table_info}
    Question: {input}

    Given an input question, first create a syntactically correct {dialect} query to run.
    Always limit your query to at most {top_k} results.

    Relevant pieces of previous conversation:
    {history}

    (You do not need to use these pieces of information if not relevant)
    """
    valid_decider_prompt_str = """Given the below input question and list of potential tables, output a comma separated list of the table names that may be necessary to answer this question.

    History: {history}

    Question: {query}

    Table Names: {table_names}

    Relevant Table Names:"""

    valid_query_prompt = PromptTemplate(
        input_variables=["input", "table_info", "dialect", "top_k", "history"],
        template=valid_query_prompt_str,
    )
    valid_decider_prompt = PromptTemplate(
        input_variables=["query", "table_names", "history"],
        template=valid_decider_prompt_str,
        output_parser=CommaSeparatedListOutputParser(),
    )

    db = SQLDatabase.from_uri("sqlite:///:memory:")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    decider_memory = ConversationBufferMemory(memory_key="history", input_key="query")
    query_memory = ConversationBufferMemory(memory_key="history", input_key="input")
    db_chain = SQLDatabaseSequentialChain.from_llm(
        llm,
        db,
        decider_memory=decider_memory,
        query_memory=query_memory,
        decider_prompt=valid_decider_prompt,
        query_prompt=valid_query_prompt,
        verbose=True,
    )
    assert db_chain.run("hello") == "Final Answer: baz"
