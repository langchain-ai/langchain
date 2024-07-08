"""Test Kinetica Chat API wrapper."""

import logging
from typing import TYPE_CHECKING, Generator

import pandas as pd
import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.chat_models.kinetica import (
    ChatKinetica,
    KineticaSqlOutputParser,
    KineticaSqlResponse,
    KineticaUtil,
)

if TYPE_CHECKING:
    import gpudb


LOG = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    return {
        # Replace the Authorization request header with "DUMMY" in cassettes
        "filter_headers": [("authorization", "DUMMY")],
    }


class TestChatKinetica:
    """Integration tests for `Kinetica` chat models.

    You must have `gpudb`, `typeguard`, and `faker` packages installed to run these
    tests. pytest-vcr cassettes are provided for offline testing.

    For more information see https://docs.kinetica.com/7.1/sql-gpt/concepts/.

    These integration tests follow a workflow:

    1. The `test_setup()` will create a table with fake user profiles and and a related
       LLM context for the table.

    2. The LLM context is retrieved from the DB and used to create a chat prompt
       template.

    3. A chain is constructed from the chat prompt template.

    4. The chain is executed to generate the SQL and execute the query.
    """

    table_name = "demo.test_profiles"
    context_name = "demo.test_llm_ctx"
    num_records = 100

    @classmethod
    @pytest.mark.vcr()
    def test_setup(cls) -> "gpudb.GPUdb":
        """Create the connection, test table, and LLM context."""

        kdbc = KineticaUtil.create_kdbc()
        cls._create_test_table(kdbc, cls.table_name, cls.num_records)
        cls._create_llm_context(kdbc, cls.context_name)
        return kdbc

    @pytest.mark.vcr()
    def test_create_llm(self) -> None:
        """Create an LLM instance."""
        import gpudb

        kinetica_llm = ChatKinetica()  # type: ignore[call-arg]
        LOG.info(kinetica_llm._identifying_params)

        assert isinstance(kinetica_llm.kdbc, gpudb.GPUdb)
        assert kinetica_llm._llm_type == "kinetica-sqlassist"

    @pytest.mark.vcr()
    def test_load_context(self) -> None:
        """Load the LLM context from the DB."""
        kinetica_llm = ChatKinetica()  # type: ignore[call-arg]
        ctx_messages = kinetica_llm.load_messages_from_context(self.context_name)

        system_message = ctx_messages[0]
        assert isinstance(system_message, SystemMessage)

        last_question = ctx_messages[-2]
        assert isinstance(last_question, HumanMessage)
        assert last_question.content == "How many male users are there?"

    @pytest.mark.vcr()
    def test_generate(self) -> None:
        """Generate SQL from a chain."""
        kinetica_llm = ChatKinetica()  # type: ignore[call-arg]

        # create chain
        ctx_messages = kinetica_llm.load_messages_from_context(self.context_name)
        ctx_messages.append(("human", "{input}"))
        prompt_template = ChatPromptTemplate.from_messages(ctx_messages)
        chain = prompt_template | kinetica_llm

        resp_message = chain.invoke(
            {"input": "What are the female users ordered by username?"}
        )
        LOG.info(f"SQL Response: {resp_message.content}")
        assert isinstance(resp_message, AIMessage)

    @pytest.mark.vcr()
    def test_full_chain(self) -> None:
        """Generate SQL from a chain and execute the query."""
        kinetica_llm = ChatKinetica()  # type: ignore[call-arg]

        # create chain
        ctx_messages = kinetica_llm.load_messages_from_context(self.context_name)
        ctx_messages.append(("human", "{input}"))
        prompt_template = ChatPromptTemplate.from_messages(ctx_messages)
        chain = (
            prompt_template
            | kinetica_llm
            | KineticaSqlOutputParser(kdbc=kinetica_llm.kdbc)
        )
        sql_response: KineticaSqlResponse = chain.invoke(
            {"input": "What are the female users ordered by username?"}
        )

        assert isinstance(sql_response, KineticaSqlResponse)
        LOG.info(f"SQL Response: {sql_response.sql}")
        assert isinstance(sql_response.dataframe, pd.DataFrame)
        users = sql_response.dataframe["username"]
        assert users[0] == "alexander40"

    @classmethod
    def _create_fake_records(cls, count: int) -> Generator:
        """Generator for fake records."""
        import faker

        faker.Faker.seed(5467)
        faker_inst = faker.Faker(locale="en-US")
        for id in range(0, count):
            rec = dict(id=id, **faker_inst.simple_profile())
            rec["birthdate"] = pd.Timestamp(rec["birthdate"])
            yield rec

    @classmethod
    def _create_test_table(
        cls, kinetica_dbc: "gpudb.GPUdb", table_name: str, num_records: int
    ) -> "gpudb.GPUdbTable":
        """Create a table from the fake records generator."""
        import gpudb

        table_df = pd.DataFrame.from_records(
            data=cls._create_fake_records(num_records), index="id"
        )

        LOG.info(f"Creating test table '{table_name}' with {num_records} records...")
        gpudb_table = gpudb.GPUdbTable.from_df(
            table_df,
            db=kinetica_dbc,
            table_name=table_name,
            clear_table=True,
            load_data=True,
            column_types={},
        )
        return gpudb_table

    @classmethod
    def _check_error(cls, response: dict) -> None:
        """Convert a DB error into an exception."""
        status = response["status_info"]["status"]
        if status != "OK":
            message = response["status_info"]["message"]
            raise Exception("[%s]: %s" % (status, message))

    @classmethod
    def _create_llm_context(
        cls, kinetica_dbc: "gpudb.GPUdb", context_name: str
    ) -> None:
        """Create an LLM context for the table."""

        sql = f"""
        CREATE OR REPLACE CONTEXT {context_name}
        (
            TABLE = {cls.table_name}
            COMMENT = 'Contains user profiles.'
        ),
        (
            SAMPLES = (
            'How many male users are there?' = 
            'select count(1) as num_users
            from {cls.table_name}
            where sex = ''M'';')
        )
        """
        LOG.info(f"Creating context: {context_name}")
        response = kinetica_dbc.execute_sql(sql)
        cls._check_error(response)
