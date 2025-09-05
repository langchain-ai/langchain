from langchain_anthropic import ChatAnthropic
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

LONG_PROMPT = """
Please be a helpful assistant.

""" + "a" * (100 * 60)  # 100 chars per line * 60 lines

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-20250514"),
    tools=[],
    prompt=LONG_PROMPT,
    middleware=[AnthropicPromptCachingMiddleware(type="ephemeral", ttl="5m", min_messages_to_cache=3)],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "abc"}}

agent.invoke({"messages": [HumanMessage("Hello")]}, config)
agent.invoke({"messages": [HumanMessage("Hello")]}, config)
result3 = agent.invoke({"messages": [HumanMessage("Hello")]}, config)


for msg in result3["messages"]:
    msg.pretty_print()

    if isinstance(msg, AIMessage):
        print(f"usage: {msg.response_metadata['usage']}")
