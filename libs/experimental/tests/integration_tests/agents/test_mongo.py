from langchain_experimental.agents import create_mongo_agent
from langchain_experimental.agents.agent_toolkits import MongoDatabaseToolkit

from langchain_experimental.utilities.mongo_database import MongoDatabase
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_create_mongo_agent() -> None:
    db = MongoDatabase.from_uri(
        "mongodb://%2Ftmp%2Fmongodb-27017.sock/test_db?inMemory=true"
    )
    queries = {"foo": "Final Answer: baz"}
    llm = FakeLLM(queries=queries, sequential_responses=True)
    toolkit = MongoDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_mongo_agent(
        llm=llm,
        toolkit=toolkit,
    )

    assert agent_executor.run("hello") == "baz"
