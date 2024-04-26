from langchain_cli.namespaces.migrate.generate.langchain import (
    generate_simplified_migrations,
)


def test_create_json_agent_migration() -> None:
    """Test the migration of create_json_agent from langchain to langchain_community."""
    raw_migrations = generate_simplified_migrations()
    json_agent_migrations = [
        migration for migration in raw_migrations if "create_json_agent" in migration[0]
    ]
    assert json_agent_migrations == [
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
    ]
