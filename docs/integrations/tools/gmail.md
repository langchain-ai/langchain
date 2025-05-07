from langchain.langgraph.prebuilt import create_react_agent

tools = ["tool_1", "tool_2"]
llm = OpenAI(model="gpt-3.5-turbo")
agent = create_react_agent(tools, llm)
response = agent.invoke("What is the current price of Tesla stock?")
print(response)

