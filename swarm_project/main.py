from langgraph_swarm import create_swarm
from langgraph.checkpoint.memory import MemorySaver
from src.agents import AGENT_DICT
from src.utils import mission_control_log
from dotenv import load_dotenv

load_dotenv()

# Initialize Swarm
memory = MemorySaver()
swarm = create_swarm(AGENT_DICT, default_active_agent="architect")

def run_mission(objective: str):
    config = {"configurable": {"thread_id": "swarm_01"}, "recursion_limit": 20}
    inputs = {"messages": [("user", objective)], "iterations": 0}
    
    print(f"🚀 MISSION STARTED: {objective}\n" + "="*50)
    
    for event in swarm.stream(inputs, config, stream_mode="values"):
        # Custom logging for 'Visual Mission Control'
        mission_control_log(event)

if __name__ == "__main__":
    run_mission("Create a Python script that calculates Fibonacci and hand off to tester.")
