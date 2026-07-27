import asyncio

from content_growth_agent.core.memory_backends.in_memory import InMemoryMemory
from content_growth_agent.core.base_memory import ChatMessage


def test_inmemory_add_get_clear():
    mem = InMemoryMemory()

    async def run():
        await mem.add_message(ChatMessage(role="user", content="hello"))
        await mem.add_message(ChatMessage(role="assistant", content="hi"))
        history = await mem.get_history()
        assert len(history) == 2
        assert history[0].content == "hello"
        await mem.clear()
        history2 = await mem.get_history()
        assert len(history2) == 0

    asyncio.run(run())
