import asyncio
from typing import Any, Dict, List

from fastapi_poe import PoeHandler

from langchain.callbacks.base import BaseCallbackHandler, CallbackManager
from langchain.chains.base import Chain
from langchain.schema import BaseLanguageModel


class LangChainFastAPIPoeHandler(PoeHandler):
    def __init__(self, chain: Chain, llm: BaseLanguageModel):
        self.chain = chain
        self.llm = llm

    async def get_response(self, query):
        callback_handler = StreamCallbackHandler()
        callback_manager = CallbackManager(
            [callback_handler, *self.llm.callback_manager.callbacks]
        )

        # TODO we need proper concurrency support here
        current_callback_manager = self.llm.callback_manager
        self.llm.callback_manager = callback_manager

        run = asyncio.create_task(self.chain.arun(query))

        async for token in callback_handler.stream():
            yield token

        await run

        self.llm.callback_manager = current_callback_manager


class StreamCallbackHandler(BaseCallbackHandler):
    @property
    def always_verbose(self):
        return True

    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]):
        # If two calls are made in a row, this resets the state
        # If two calls are made in parallel, we're in trouble...
        self.done.clear()
        self.queue = asyncio.Queue()

    def on_llm_new_token(self, token: str):
        self.queue.put_nowait(token)

    def on_llm_end(self, serialized: Dict[str, Any], prompts: List[str]):
        self.done.set()

    async def stream(self):
        while not self.done.is_set():
            token = await self.queue.get()
            yield token
