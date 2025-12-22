from langgraph_swarm import create_handoff_tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.tools.developer import write_code
from src.tools.testing import run_test

llm = ChatOpenAI(model="gpt-4o")

# 1. Define Handoff Tools
go_to_tester = create_handoff_tool(agent_name="tester")
go_to_architect = create_handoff_tool(agent_name="architect")

# 2. Architect Agent
architect = create_react_agent(
    llm,
    tools=[write_code, go_to_tester],
    state_modifier="You are the Architect. Write code and hand off to the Tester. Don't repeat failures."
)

# 3. Tester Agent
tester = create_react_agent(
    llm,
    tools=[run_test, go_to_architect],
    state_modifier="You are the Tester. If code fails, hand back to Architect with logs. If it passes, say 'MISSION_COMPLETE'."
)

# Registry for the swarm
AGENT_DICT = {"architect": architect, "tester": tester}
