from langchain.agents.middleware_agent import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.middleware.summarization import SummarizationMiddleware


agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini"),
    system_prompt="You are a helpful assistant. Please reply nicely.",
    middleware=[
        SummarizationMiddleware(
            model=ChatOpenAI(model="gpt-4.1-mini"), messages_to_keep=3
        )
    ],
)
agent = agent.compile(checkpointer=InMemorySaver())

config: RunnableConfig = {"configurable": {"thread_id": "long_convo"}}

config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "my favorite food is pizza"}, config)
agent.invoke({"messages": "my favorite color is blue"}, config)
agent.invoke({"messages": "my favorite animal is a dog"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

for msg in final_response["messages"]:
    msg.pretty_print()

"""
================================ System Message ================================

## Previous conversation summary:
User name: Bob. User's favorite food is pizza. User's favorite color is blue.
================================ Human Message =================================

my favorite animal is a dog
================================== Ai Message ==================================

Dogs are wonderful companions, Bob! Do you have a favorite breed, or maybe a dog of your own?
================================ Human Message =================================

what's my name?
================================== Ai Message ==================================

Your name is Bob! How can I assist you today, Bob?
"""
