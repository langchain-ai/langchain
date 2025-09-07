from langchain.agents import create_agent
from langchain.agents.middleware.swarm import SwarmMiddleware
from langchain.chat_models.fake import FakeToolCallingModel

tool_calls = [[{"args": {}, "id": "1", "name": "handoff_to_foo2"}], []]
model = FakeToolCallingModel(tool_calls=tool_calls)
subagents = [
    {"name": "foo1", "description": "bar1", "prompt": "hi", "tools": []},
    {"name": "foo2", "description": "bar1", "prompt": "bye", "tools": []}
]
middleware = SwarmMiddleware(agents=subagents, starting_agent="foo1")
agent = create_agent(model, [], middleware=[middleware])
print(agent.get_graph())
for s in agent.stream({"messages": [{"role": "user", "content": "hi"}]}, stream_mode="debug"):
    print(s)
