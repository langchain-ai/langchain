from typing import Sequence, List, Optional

from langchain.automaton.typedefs import MessageLike
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableLambda, Runnable


def test_router() -> None:
    messages = [AIMessage(content="Hello, world!")]

    program_state = RunnableLambda(lambda x: x)

    def route(messages: Sequence[MessageLike]) -> Optional[Runnable]:
        if isinstance(messages[-1], HumanMessage):
            return None
        else:
            return program_state

    router = RunnableLambda(route)
    assert router.invoke(messages) == []
