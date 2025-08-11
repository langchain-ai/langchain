import pytest
from langchain._api import suppress_langchain_deprecation_warning as sup2
from langchain_core._api import suppress_langchain_deprecation_warning as sup1

from langchain_cli.namespaces.migrate.generate.generic import (
    generate_simplified_migrations,
)


@pytest.mark.xfail(reason="Unknown reason")
def test_create_json_agent_migration() -> None:
    """Test the migration of create_json_agent from langchain to langchain_community."""
    with sup1(), sup2():
        raw_migrations = generate_simplified_migrations(
            from_package="langchain",
            to_package="langchain_community",
        )
        json_agent_migrations = [
            migration
            for migration in raw_migrations
            if "create_json_agent" in migration[0]
        ]
        if json_agent_migrations != [
            (
                "langchain.agents.create_json_agent",
                "langchain_community.agent_toolkits.create_json_agent",
            ),
            (
                "langchain.agents.agent_toolkits.create_json_agent",
                "langchain_community.agent_toolkits.create_json_agent",
            ),
            (
                "langchain.agents.agent_toolkits.json.base.create_json_agent",
                "langchain_community.agent_toolkits.create_json_agent",
            ),
        ]:
            msg = "json_agent_migrations did not match the expected value"
            raise ValueError(msg)


@pytest.mark.xfail(reason="Unknown reason")
def test_create_single_store_retriever_db() -> None:
    """Test migration from langchain to langchain_core."""
    with sup1(), sup2():
        raw_migrations = generate_simplified_migrations(
            from_package="langchain",
            to_package="langchain_core",
        )
        # SingleStore was an old name for VectorStoreRetriever
        single_store_migration = [
            migration for migration in raw_migrations if "SingleStore" in migration[0]
        ]
        if single_store_migration != [
            (
                "langchain.vectorstores.singlestoredb.SingleStoreDBRetriever",
                "langchain_core.vectorstores.VectorStoreRetriever",
            ),
        ]:
            msg = (
                "Unexpected migration: single_store_migration does not match expected "
                "value"
            )
            raise ValueError(msg)
