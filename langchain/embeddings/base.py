"""Interface for embedding models."""
import asyncio
import warnings
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, List, Sequence

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForEmbeddingsRun,
    CallbackManager,
    CallbackManagerForEmbeddingsRun,
    Callbacks,
)


class Embeddings(ABC):
    """Interface for embedding models."""

    _new_arg_supported: bool = False
    _expects_other_args: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.embed_documents != Embeddings.embed_documents:
            warnings.warn(
                "Embedding models must implement abstract `_embed_documents` method"
                " instead of `embed_documents`",
                DeprecationWarning,
            )
            swap = cls.embed_documents
            cls.embed_documents = Embeddings.embed_documents  # type: ignore[assignment]
            cls._embed_documents = swap  # type: ignore[assignment]
        if (
            hasattr(cls, "aembed_documents")
            and cls.aembed_documents != Embeddings.aembed_documents
        ):
            warnings.warn(
                "Embedding models must implement abstract `_aembed_documents` method"
                " instead of `aembed_documents`",
                DeprecationWarning,
            )
            aswap = cls.aembed_documents
            cls.aembed_documents = (  # type: ignore[assignment]
                Embeddings.aembed_documents
            )
            cls._aembed_documents = aswap  # type: ignore[assignment]
        parameters = signature(cls._embed_documents).parameters
        cls._new_arg_supported = parameters.get("run_manager") is not None
        cls._expects_other_args = (not cls._new_arg_supported) and len(parameters) > 1

    @abstractmethod
    def _embed_documents(
        self,
        texts: List[str],
        *,
        run_managers: Sequence[CallbackManagerForEmbeddingsRun],
        **kwargs: Any
    ) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def _embed_query(
        self, text: str, *, run_manager: CallbackManagerForEmbeddingsRun, **kwargs: Any
    ) -> List[float]:
        """Embed query text."""

    @abstractmethod
    async def _aembed_documents(
        self,
        texts: List[str],
        *,
        run_managers: Sequence[AsyncCallbackManagerForEmbeddingsRun],
        **kwargs: Any
    ) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    async def _aembed_query(
        self,
        text: str,
        *,
        run_manager: AsyncCallbackManagerForEmbeddingsRun,
        **kwargs: Any
    ) -> List[float]:
        """Embed query text."""

    def embed_documents(
        self, texts: List[str], *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[List[float]]:
        """Embed search docs."""

        callback_manager = CallbackManager.configure(
            callbacks, None, verbose=kwargs.get("verbose", False)
        )
        run_managers = callback_manager.on_embeddings_start(
            texts,
            **kwargs,
        )
        try:
            if self._new_arg_supported:
                result = self._embed_documents(
                    texts, run_managers=run_managers, **kwargs
                )
            elif self._expects_other_args:
                result = self._embed_documents(texts, **kwargs)
            else:
                result = self._embed_documents(texts)  # type: ignore[call-arg]
        except Exception as e:
            for run_manager in run_managers:
                run_manager.on_embeddings_error(e)
            raise e
        else:
            for run_manager in run_managers:
                run_manager.on_embeddings_end(
                    result,
                    **kwargs,
                )
            return result

    def embed_query(
        self, text: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[float]:
        """Embed query text."""
        from langchain.callbacks.manager import CallbackManager

        callback_manager = CallbackManager.configure(
            callbacks, None, verbose=kwargs.get("verbose", False)
        )
        run_managers = callback_manager.on_embeddings_start(
            [text],
            **kwargs,
        )
        try:
            if self._new_arg_supported:
                result = self._embed_query(text, run_manager=run_managers[0], **kwargs)
            elif self._expects_other_args:
                result = self._embed_query(text, **kwargs)
            else:
                result = self._embed_query(text)  # type: ignore[call-arg]
        except Exception as e:
            run_managers[0].on_embeddings_error(e)
            raise e
        else:
            run_managers[0].on_embeddings_end(
                result,
                **kwargs,
            )
            return result

    async def aembed_documents(
        self, texts: List[str], *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[List[float]]:
        """Asynchronously embed search docs."""

        callback_manager = AsyncCallbackManager.configure(
            callbacks, None, verbose=kwargs.get("verbose", False)
        )
        run_managers = await callback_manager.on_embeddings_start(
            texts,
            **kwargs,
        )
        try:
            if self._new_arg_supported:
                result = await self._aembed_documents(
                    texts, run_managers=run_managers, **kwargs
                )
            elif self._expects_other_args:
                result = await self._aembed_documents(texts, **kwargs)
            else:
                result = await self._aembed_documents(texts)  # type: ignore[call-arg]
        except Exception as e:
            tasks = [run_manager.on_embeddings_error(e) for run_manager in run_managers]
            await asyncio.gather(*tasks)
            raise e
        else:
            tasks = [
                run_manager.on_embeddings_end(
                    results,
                    **kwargs,
                )
                for run_manager, results in zip(run_managers, result)
            ]
            await asyncio.gather(*tasks)
            return result

    async def aembed_query(
        self, text: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[float]:
        """Asynchronously embed query text."""
        from langchain.callbacks.manager import AsyncCallbackManager

        callback_manager = AsyncCallbackManager.configure(
            callbacks, None, verbose=kwargs.get("verbose", False)
        )
        run_managers = await callback_manager.on_embeddings_start(
            [text],
            **kwargs,
        )
        try:
            if self._new_arg_supported:
                result = await self._aembed_query(
                    text, run_manager=run_managers[0], **kwargs
                )
            elif self._expects_other_args:
                result = await self._aembed_query(text, **kwargs)
            else:
                result = await self._aembed_query(text)  # type: ignore[call-arg]
        except Exception as e:
            await run_managers[0].on_embeddings_error(e)
            raise e
        else:
            await run_managers[0].on_embeddings_end(
                result,
                **kwargs,
            )
            return result
