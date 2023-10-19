import time
from typing import Any, Callable, List

from langchain.prompts.chat import (
    BaseChatPromptTemplate,
)
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.tools.base import BaseTool

from langchain_experimental.autonomous_agents.autogpt.prompt_generator import get_prompt
from langchain_experimental.pydantic_v1 import BaseModel


# This class has a metaclass conflict: both `BaseChatPromptTemplate` and `BaseModel`
# define a metaclass to use, and the two metaclasses attempt to define
# the same functions but in mutually-incompatible ways.
# It isn't clear how to resolve this, and this code predates mypy
# beginning to perform that check.
#
# Mypy errors:
# ```
# Definition of "__private_attributes__" in base class "BaseModel" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__repr_name__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__pretty__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__repr_str__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Definition of "__rich_repr__" in base class "Representation" is
#   incompatible with definition in base class "BaseModel"  [misc]
# Metaclass conflict: the metaclass of a derived class must be
#   a (non-strict) subclass of the metaclasses of all its bases  [misc]
# ```
#
# TODO: look into refactoring this class in a way that avoids the mypy type errors
class AutoGPTPrompt(BaseChatPromptTemplate, BaseModel):  # type: ignore[misc]
    """Prompt for AutoGPT."""

    ai_name: str
    ai_role: str
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    def construct_full_prompt(self, goals: List[str]) -> str:
        prompt_start = (
            "Your decisions must always be made independently "
            "without seeking user assistance.\n"
            "Play to your strengths as an LLM and pursue simple "
            "strategies with no legal complications.\n"
            "If you have completed all your tasks, make sure to "
            'use the "finish" command.'
        )
        # Construct full prompt
        full_prompt = (
            f"You are {self.ai_name}, {self.ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        )
        for i, goal in enumerate(goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n\n{get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs["goals"]))
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}"
        )
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(
            time_prompt.content
        )
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory]
        )
        while used_tokens + relevant_memory_tokens > 2500:
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory]
            )
        content_format = (
            f"This reminds you of these events "
            f"from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages
