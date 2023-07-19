import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert

from langchain import SQLDatabase
from langchain.agents import AgentExecutor, AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from tests.unit_tests.llms.fake_llm import FakeLLM

metadata_obj = MetaData()

user = Table(
    "user",
    metadata_obj,
    Column("user_id", Integer, primary_key=True),
    Column("user_name", String(16), nullable=False),
    Column("user_company", String(16), nullable=False),
)


@pytest.fixture(scope="module")
def db() -> SQLDatabase:
    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)
    stmt = insert(user).values(user_id=13, user_name="Harrison", user_company="Foo")
    with engine.connect() as conn:
        conn.execute(stmt)
        conn.commit()
    db = SQLDatabase(engine)
    return db


@pytest.fixture(scope="module")
def sql_agent(db: SQLDatabase) -> AgentExecutor:
    llm = ChatOpenAI(
        temperature=0.5, max_tokens=1000, model_name="gpt-3.5-turbo", verbose=True
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )


def test_sql_agent_run(sql_agent: AgentExecutor) -> None:
    result = sql_agent.run("What is the name of the user with id 13?")
    assert result == "Harrison"


@pytest.mark.asyncio
async def test_sql_agent_arun(sql_agent: AgentExecutor) -> None:
    result = sql_agent.run("What is the name of the user with id 13?")
    assert result == "Harrison"
