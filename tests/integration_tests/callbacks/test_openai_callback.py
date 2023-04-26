import asyncio

import pytest

from langchain import OpenAI
from langchain.callbacks import get_openai_callback


@pytest.mark.asyncio
async def test_openai_callback():
    llm = OpenAI(temperature=0)
    with get_openai_callback() as cb:
        llm("What is the square root of 4?")

    total_tokens = cb.total_tokens
    assert total_tokens > 0

    with get_openai_callback() as cb:
        llm("What is the square root of 4?")
        llm("What is the square root of 4?")

    assert cb.total_tokens == total_tokens * 2

    with get_openai_callback() as cb:
        await asyncio.gather(
            *[llm.agenerate(["What is the square root of 4?"]) for _ in range(3)]
        )

    assert cb.total_tokens == total_tokens * 3

    task = asyncio.create_task(llm.agenerate(["What is the square root of 4?"]))
    with get_openai_callback() as cb:
        await llm.agenerate(["What is the square root of 4?"])

    await task
    assert cb.total_tokens == total_tokens
