import asyncio
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler, CallbackManager
from langchain.chains.base import Chain
from langchain.schema import BaseLanguageModel


class LangChainPoeHandler(PoeHandler):
    def __init__(self, chain: Chain, llm: BaseLanguageModel):
        self.chain = chain
        self.llm = llm

    async def get_response(self, query):
        callback_handler = PoeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        self.llm.callback_manager = callback_manager

        run = asyncio.create_task(self.chain.acall(query))

        while not callback_handler.done.is_set():
            token = await callback_handler.queue.get()
            yield token

        await run


class PoeCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]):
        pass

    def on_llm_new_token(self, token: str):
        self.queue.put_nowait(token)

    def on_llm_end(self, serialized: Dict[str, Any], prompts: List[str]):
        self.done.set()
