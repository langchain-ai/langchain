from langchain.agents import create_agent
from langchain.agents.middleware.deepagents import DeepAgentMiddleware
from langchain.chat_models.fake import FakeToolCallingModel

model = FakeToolCallingModel()
agent = create_agent(model, [], middleware=[DeepAgentMiddleware()])

for s in agent.stream({"messages": [{"role": "user", "content": "hi"}]}, stream_mode="debug"):
    print(s)
