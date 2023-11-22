from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish


def test_lc_namespace() -> None:
    assert AgentAction.get_lc_namespace() == [
        "langchain",
        "schema",
        "agent",
    ]
    assert AgentActionMessageLog.get_lc_namespace() == [
        "langchain",
        "schema",
        "agent",
    ]
    assert AgentFinish.get_lc_namespace() == [
        "langchain",
        "schema",
        "agent",
    ]
