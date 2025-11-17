from langchain.agents.factory import create_agent
from langchain_core.messages import AIMessage, HumanMessage


class DummyModel:
    """Minimal fake model that returns an AIMessage without a name."""

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        return AIMessage(content="dummy response")


class NamedDummyModel(DummyModel):
    def invoke(self, messages):
        return AIMessage(content="named response", name="ExplicitName")


def test_agent_propagates_name_by_default():
    agent = create_agent(model=DummyModel(), tools=[], name="TestAgentName")
    # invoke graph with a user message
    resp = agent.invoke({"messages": [HumanMessage(content="hi")]})
    # find AIMessage and ensure its name was set to agent name
    ai_messages = [m for m in resp["messages"] if isinstance(m, AIMessage)]
    assert ai_messages, "expected at least one AIMessage in response"
    assert any(m.name == "TestAgentName" for m in ai_messages)


def test_agent_preserves_explicit_message_name():
    agent = create_agent(model=NamedDummyModel(), tools=[], name="AgentShouldNotOverride")
    resp = agent.invoke({"messages": [HumanMessage(content="hello")]})
    ai_messages = [m for m in resp["messages"] if isinstance(m, AIMessage)]
    assert ai_messages
    # explicit message name from model should be preserved
    assert any(m.name == "ExplicitName" for m in ai_messages)
