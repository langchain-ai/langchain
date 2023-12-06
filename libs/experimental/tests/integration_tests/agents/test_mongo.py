from langchain_experimental.agents.agent_toolkits import (
    MongoDatabaseToolkit,
    create_mongo_agent,
)
from langchain_experimental.utilities import MongoDatabase
from tests.unit_tests.fake_llm import FakeLLM


def test_create_mongo_agent() -> None:
    db = MongoDatabase.from_uri("mongodb://localhost:27017/test_db")
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    toolkit = MongoDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_mongo_agent(
        llm=llm,
        toolkit=toolkit,
    )

    assert agent_executor.run("hello") == "baz"
