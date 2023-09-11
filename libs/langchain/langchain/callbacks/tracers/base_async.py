import asyncio
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run
from langchain.schema.document import Document
from langchain.schema.output import ChatGenerationChunk, GenerationChunk, LLMResult


class InnerSyncTracer(BaseTracer):
    def _persist_run(self, run: Run) -> None:
        """The Langchain Tracer uses Post/Patch rather than persist."""


class AsyncBaseTracer(AsyncCallbackHandler, ABC):
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.inner_tracer = InnerSyncTracer()

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        async with self.lock:
            llm_run = self.inner_tracer.on_llm_start(serialized, prompts, **kwargs)

        await asyncio.gather(
            self._on_llm_start(llm_run),
            self._on_run_create(llm_run),
            return_exceptions=True,
        )

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self.inner_tracer.on_llm_new_token(token, chunk=chunk, **kwargs)

        await self._on_llm_new_token(llm_run, token, chunk)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        async with self.lock:
            llm_run = self.inner_tracer.on_llm_end(response, **kwargs)

        await asyncio.gather(
            self._on_llm_end(llm_run),
            self._on_run_update(llm_run),
            return_exceptions=True,
        )

    async def on_llm_error(
        self, error: Exception | KeyboardInterrupt, **kwargs: Any
    ) -> None:
        async with self.lock:
            llm_run = self.inner_tracer.on_llm_error(error, **kwargs)

        await asyncio.gather(
            self._on_llm_error(llm_run),
            self._on_run_update(llm_run),
            return_exceptions=True,
        )

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            chain_run = self.inner_tracer.on_chain_start(serialized, inputs, **kwargs)

        await asyncio.gather(
            self._on_chain_start(chain_run),
            self._on_run_create(chain_run),
            return_exceptions=True,
        )

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            chain_run = self.inner_tracer.on_chain_end(outputs, **kwargs)

        await asyncio.gather(
            self._on_chain_end(chain_run),
            self._on_run_update(chain_run),
            return_exceptions=True,
        )

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            chain_run = self.inner_tracer.on_chain_error(error, **kwargs)

        await asyncio.gather(
            self._on_chain_error(chain_run),
            self._on_run_update(chain_run),
            return_exceptions=True,
        )

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            tool_run = self.inner_tracer.on_tool_start(serialized, input_str, **kwargs)

        await asyncio.gather(
            self._on_tool_start(tool_run),
            self._on_run_create(tool_run),
            return_exceptions=True,
        )

    async def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            tool_run = self.inner_tracer.on_tool_end(output, **kwargs)

        await asyncio.gather(
            self._on_tool_end(tool_run),
            self._on_run_update(tool_run),
            return_exceptions=True,
        )

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            tool_run = self.inner_tracer.on_tool_error(error, **kwargs)

        await asyncio.gather(
            self._on_tool_error(tool_run),
            self._on_run_update(tool_run),
            return_exceptions=True,
        )

    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            retriever_run = self.inner_tracer.on_retriever_start(
                serialized, query, **kwargs
            )

        await asyncio.gather(
            self._on_retriever_start(retriever_run),
            self._on_run_create(retriever_run),
            return_exceptions=True,
        )

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            retriever_run = self.inner_tracer.on_retriever_end(documents, **kwargs)

        await asyncio.gather(
            self._on_retriever_end(retriever_run),
            self._on_run_update(retriever_run),
            return_exceptions=True,
        )

    async def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            retriever_run = self.inner_tracer.on_retriever_error(error, **kwargs)

        await asyncio.gather(
            self._on_retriever_error(retriever_run),
            self._on_run_update(retriever_run),
            return_exceptions=True,
        )

    async def _on_run_create(self, run: Run) -> None:
        """Process the Run upon creation."""

    async def _on_run_update(self, run: Run) -> None:
        """Process the Run upon update."""

    async def _on_llm_start(self, run: Run) -> None:
        """Process the LLM Run upon start."""

    async def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]],
    ) -> None:
        """Process new LLM token."""

    async def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""

    async def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""

    async def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""

    async def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""

    async def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""

    async def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""

    async def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""

    async def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""

    async def _on_chat_model_start(self, run: Run) -> None:
        """Process the Chat Model Run upon start."""

    async def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start."""

    async def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run."""

    async def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error."""
