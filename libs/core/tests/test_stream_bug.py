import asyncio
from typing import Any
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers.openai_tools import PydanticToolsParser


class Person(BaseModel):
    name: str
    age: int


class NonStreamingModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "Person", "args": {"name": "Ada", "age": 36}, "id": "call_1"}],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "non-streaming"


chain = NonStreamingModel() | PydanticToolsParser(tools=[Person], first_tool_only=True)
print("invoke :", chain.invoke("hi"))
print("stream :", list(chain.stream("hi")))


async def main():
    return [c async for c in chain.astream("hi")]


print("astream:", asyncio.run(main()))