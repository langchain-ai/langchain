import asyncio

from typing_extensions import Unpack

from langchain.schema.runnable import RunnableConfig, Runnable


class FakeRunnable(Runnable[str, int]):
    def invoke(
        self,
        input: str,
        **kwargs: Unpack[RunnableConfig],
    ) -> int:
        return len(input)


fake = FakeRunnable()

print("invoke", fake.invoke("hello"))

for token in fake.stream("hello"):
    print("stream", token)

print("batch", fake.batch(["hello", "world"]))


async def test_async() -> None:
    print("ainvoke", await fake.ainvoke("hello"))
    async for token in fake.astream("hello"):
        print("astream", token)
    print("abatch", await fake.abatch(["hello", "world"]))


asyncio.run(test_async())
