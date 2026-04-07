"""CognitiveSynergyAgent"""

from langchain.agents import AgentExecutor
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from typing import List, Dict, Any
import time

class MultiMemory:
    def __init__(self):
        self.working_memory: List[BaseMessage] = []
        self.declarative: Dict[str, Any] = {}
        self.episodic: List[Dict] = []
        self.procedural: List[Dict] = []
        self.attention_scores: Dict[str, float] = {}

    def add_to_working(self, item):
        self.working_memory.append(item)
        if len(self.working_memory) > 20:
            self.working_memory.pop(0)

    def add_episodic(self, event: Dict):
        self.episodic.append(event)
        self.attention_scores[str(len(self.episodic))] = 1.0

    def retrieve_procedural(self, goal: str) -> List[Dict]:
        return [r for r in self.procedural if goal.lower() in r.get("condition", "").lower()]

class CognitiveSynergyAgent(AgentExecutor):
    def __init__(self, llm: Runnable, tools: List, **kwargs):
        super().__init__(llm=llm, tools=tools, **kwargs)
        self.memory = MultiMemory()
        self.current_goal = None

    def cognitive_cycle(self, input: str) -> str:
        self.memory.add_to_working({"role": "user", "content": input})
        rules = self.memory.retrieve_procedural(self.current_goal or input)
        episodic = self.memory.episodic[-3:] if self.memory.episodic else []
        context = f"GOAL: {self.current_goal or input}\nWM: {self.memory.working_memory[-5:]}\nEP: {episodic}"
        response = self.llm.invoke(context + "\nReason step-by-step. Next action or subgoal?")
        if "subgoal" in response.content.lower():
            self.current_goal = response.content.split("subgoal:")[-1].strip()
            self.memory.add_episodic({"type": "subgoal", "content": self.current_goal})
        else:
            result = super().invoke({"input": response.content})
            self.memory.add_episodic({"type": "action_result", "content": str(result)})
        reflection = self.llm.invoke(f"Reflect. Improvements? Episodes: {len(self.memory.episodic)}")
        self.memory.add_episodic({"type": "reflection", "content": reflection.content})
        return response.content

    def run_autonomous_loop(self, goal: str, max_steps: int = 20):
        self.current_goal = goal
        for step in range(max_steps):
            print(f"🔄 Step {step+1}/{max_steps} - Goal: {self.current_goal}")
            result = self.cognitive_cycle(self.current_goal)
            print(result)
            time.sleep(0.5)
            if "goal achieved" in result.lower():
                break
        return "Loop completed."

def create_haystack_pipeline(retriever, generator):
    return {"name": "haystack_rag", "func": lambda q: generator.invoke(retriever.invoke(q))}
