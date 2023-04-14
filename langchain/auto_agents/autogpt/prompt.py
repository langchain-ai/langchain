from typing import List, Any
from pydantic import BaseModel, Field
from langchain.prompts.chat import BaseChatPromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatMessagePromptTemplate
from langchain.auto_agents.autogpt.prompt_generator import get_prompt
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, ChatMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever
import time


class AutoGPTPrompt(BaseChatPromptTemplate, BaseModel):

    ai_name: str
    ai_role: str
    tools: List[BaseTool]
    ai_goals: List[str] = Field(default=["Increase net worth", "Grow Twitter Account",
                    "Develop and manage multiple businesses autonomously"])

    def construct_full_prompt(self) -> str:
        """
        Returns a prompt to the user with the class information in an organized fashion.

        Parameters:
            None

        Returns:
            full_prompt (str): A string containing the initial prompt for the user including the ai_name, ai_role and ai_goals.
        """

        prompt_start = """Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications."""

        # Construct full prompt
        full_prompt = f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        messages = []
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_memory = memory.get_relevant_documents(str(previous_messages[-2:]))
        messages.append(SystemMessage(content=self.construct_full_prompt()))
        messages.append(SystemMessage(content=f"The current time and date is {time.strftime('%c')}"))
        messages.append(SystemMessage(content=f"This reminds you of these events from your past:\n{relevant_memory}\n\n"))
        messages.extend(previous_messages[-2:])
        messages.append(HumanMessage(content=kwargs["user_input"]))
        return messages




