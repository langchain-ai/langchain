"""
Orchestrator

This file connects the router and the memory.
It receives the incoming message from the user, checks the memory, routes to the appropriate department, and generates the response.
"""
from router import get_routing_prompt
from memory import SustainableMemory

class SustainableAgent:
    def __init__(self):
        self.memory = SustainableMemory()
        self.router_prompt = get_routing_prompt()
        
    def process_request(self, chat_history: str, user_input: str) -> str:
        # 1. Retrieve past lessons from memory (Runs the Hybrid Search algorithm)
        past_lessons = self.memory.get_relevant_lessons(user_input)
        
        # 2. Add lessons to the Context
        context_warning = ""
        if past_lessons:
            context_warning = "\n\n[SYSTEM WARNING]: You have made the following mistakes in the past, do not repeat them:\n"
            for lesson in past_lessons:
                context_warning += f"- Topic: {lesson.topic} | Past Mistake: {lesson.mistake} | Correction: {lesson.correction}\n"
                
        # 3. Prepare the Router
        # The prompt is generated using the defined rules
        formatted_prompt = self.router_prompt.format(
            chat_history=chat_history, 
            user_input=user_input + context_warning
        )
        
        # In a production environment, invoke the LLM here.
        # Example: return self.llm.invoke(formatted_prompt)
        
        print(f"[DEBUG] Prepared LLM Prompt:\n{formatted_prompt}")
        return "Task dispatched successfully."
