from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for solving math expressions"
    )
]

llm = ChatOpenAI(temperature=0)

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

response = agent.run("What is 45 * 3?")
print(response)