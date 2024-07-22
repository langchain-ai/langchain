from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from langchain_robocorp import ActionServerToolkit

# Initialize LLM chat model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize Action Server Toolkit
toolkit = ActionServerToolkit(url="http://localhost:8080")
tools = toolkit.get_tools()

# Initialize Agent
system_message = SystemMessage(content="You are a helpful assistant")
prompt = OpenAIFunctionsAgent.create_prompt(system_message)
agent = OpenAIFunctionsAgent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

# Initialize Agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Typings for Langserve playground
class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: str


agent_executor = agent_executor.with_types(input_type=Input, output_type=Output)  # type: ignore[arg-type, assignment]
