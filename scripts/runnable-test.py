import asyncio

from typing_extensions import Unpack

from langchain.schema.runnable import RunnableConfig, Runnable

from langchain.llms.openai import OpenAI

from langchain.prompts.base import StringPromptValue


class FakeRunnable(Runnable[str, int]):
    def invoke(
        self,
        input: str,
        **kwargs: Unpack[RunnableConfig],
    ) -> int:
        return len(input)


fake = FakeRunnable()

llm = OpenAI()

print("invoke fake", fake.invoke("hello"))
for ftoken in fake.stream("hello"):
    print("stream fake", ftoken)
print("batch fake", fake.batch(["hello", "world"]))

print("invoke llm", llm.invoke("say hi"))
for ltoken in llm.stream("say hi"):
    print("stream llm", ltoken)
print(
    "batch llm",
    llm.batch(["say hi", StringPromptValue(text="say hello")]),
)


async def test_async() -> None:
    print("ainvoke", await fake.ainvoke("say hi"))
    async for ftoken in fake.astream("say hi"):
        print("astream", ftoken)
    print("abatch", await fake.abatch(["say hi", "say hello"]))

    print("ainvoke llm", await llm.ainvoke(StringPromptValue(text="say hi")))
    async for ltoken in llm.astream(StringPromptValue(text="say hi")):
        print("astream llm", ltoken)
    print(
        "abatch llm",
        await llm.abatch(
            [StringPromptValue(text="say hi"), StringPromptValue(text="say hello")]
        ),
    )


asyncio.run(test_async())
