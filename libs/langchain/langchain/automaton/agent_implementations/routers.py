from typing import Sequence, Optional, List

from langchain.automaton.chat_agent import Router
from langchain.automaton.typedefs import MessageLike, AgentFinish
from langchain.schema.runnable import Runnable, RunnableConfig, RunnableLambda


def create_chat_router(
    program: Runnable,
) -> Router:
    def router(
        messages: Sequence[MessageLike],
        *,
        config: Optional[RunnableConfig] = None,
    ) -> Optional[List[MessageLike]]:
        last_message = messages[-1] if messages else None
        if not last_message:
            return []

        match last_message:
            case AgentFinish():
                return []
            case _:
                return program.invoke(messages, config=config)

    return RunnableLambda(router)
