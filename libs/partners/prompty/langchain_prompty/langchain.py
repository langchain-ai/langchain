from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda

from .utils import load, prepare


def create_chat_prompt(
    path: str, input_name_agent_scratchpad: str = "agent_scratchpad"
) -> Runnable[Dict[str, Any], ChatPromptTemplate]:
    """Create a chat prompt from a Langchain schema."""

    def runnable_chat_lambda(inputs: Dict[str, Any]) -> ChatPromptTemplate:
        p = load(path)
        parsed = prepare(p, inputs)
        lc_messages = []
        for message in parsed:
            lc_messages.append((message["role"], message["content"]))

        lc_messages.append(
            MessagesPlaceholder(
                variable_name=input_name_agent_scratchpad, optional=True
            )  # type: ignore[arg-type]
        )
        lc_p = ChatPromptTemplate.from_messages(lc_messages)
        lc_p = lc_p.partial(**p.inputs)

        return lc_p

    return RunnableLambda(runnable_chat_lambda)
