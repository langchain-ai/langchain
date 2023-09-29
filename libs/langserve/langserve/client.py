from __future__ import annotations

import asyncio
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence, Union
from urllib.parse import urljoin

import httpx
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.load.dump import dumpd
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
)
from langchain.schema.runnable.utils import Input, Output

from langserve.serialization import simple_dumpd, simple_loads


def _without_callbacks(config: Optional[RunnableConfig]) -> RunnableConfig:
    """Evict callbacks from the config since those are definitely not supported."""
    _config = config or {}
    return {k: v for k, v in _config.items() if k != "callbacks"}


def _raise_for_status(response: httpx.Response) -> None:
    """Re-raise with a more informative message.

    Args:
        response: The response to check

    Raises:
        httpx.HTTPStatusError: If the response is not 2xx, appending the response
                               text to the message
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        message = str(e)
        # Append the response text if it exists, as it may contain more information
        # Especially useful when the user's request is malformed
        if e.response.text:
            message += f" for {e.response.text}"

        raise httpx.HTTPStatusError(
            message=message,
            request=e.request,
            response=e.response,
        )


def _is_async() -> bool:
    """Return True if we are in an async context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return True


def _close_clients(sync_client: httpx.Client, async_client: httpx.AsyncClient) -> None:
    """Close the async and sync clients.

    _close_clients should not be a bound method since it is called by a weakref
    finalizer.

    Args:
        sync_client: The sync client to close
        async_client: The async client to close
    """
    sync_client.close()
    if _is_async():
        # Use a ThreadPoolExecutor to run async_client_close in a separate thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit the async_client_close coroutine to the thread pool
            future = executor.submit(asyncio.run, async_client.aclose())
            future.result()
    else:
        asyncio.run(async_client.aclose())


class RemoteRunnable(Runnable[Input, Output]):
    """A RemoteRunnable is a runnable that is executed on a remote server.

    This client implements the majority of the runnable interface.

    The following features are not supported:

    - `batch` with `return_exceptions=True` since we do not support exception
      translation from the server.
    - Callbacks via the `config` argument as serialization of callbacks is not
      supported.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize the client.

        Args:
            url: The url of the server
            timeout: The timeout for requests
        """
        self.url = url
        self.sync_client = httpx.Client(base_url=url, timeout=timeout)
        self.async_client = httpx.AsyncClient(base_url=url, timeout=timeout)

        # Register cleanup handler once RemoteRunnable is garbage collected
        weakref.finalize(self, _close_clients, self.sync_client, self.async_client)

    def _invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """Invoke the runnable with the given input and config."""
        response = self.sync_client.post(
            "/invoke",
            json={
                "input": simple_dumpd(input),
                "config": _without_callbacks(config),
                "kwargs": kwargs,
            },
        )
        _raise_for_status(response)
        return simple_loads(response.text)["output"]

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        if kwargs:
            raise NotImplementedError("kwargs not implemented yet.")
        return self._call_with_config(self._invoke, input, config=config)

    async def _ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        response = await self.async_client.post(
            "/invoke",
            json={
                "input": simple_dumpd(input),
                "config": _without_callbacks(config),
                "kwargs": kwargs,
            },
        )
        _raise_for_status(response)
        return simple_loads(response.text)["output"]

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        if kwargs:
            raise NotImplementedError("kwargs not implemented yet.")
        return await self._acall_with_config(self._ainvoke, input, config)

    def _batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        if not inputs:
            return []
        if return_exceptions:
            raise NotImplementedError(
                "return_exceptions is not supported for remote clients"
            )

        if isinstance(config, list):
            _config = [_without_callbacks(c) for c in config]
        else:
            _config = _without_callbacks(config)

        response = self.sync_client.post(
            "/batch",
            json={
                "inputs": simple_dumpd(inputs),
                "config": _config,
                "kwargs": kwargs,
            },
        )
        _raise_for_status(response)
        return simple_loads(response.text)["output"]

    def batch(
        self,
        inputs: List[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[Output]:
        if kwargs:
            raise NotImplementedError("kwargs not implemented yet.")
        return self._batch_with_config(self._batch, inputs, config)

    async def _abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """Batch invoke the runnable."""
        if not inputs:
            return []
        if return_exceptions:
            raise NotImplementedError(
                "return_exceptions is not supported for remote clients"
            )

        if isinstance(config, list):
            _config = [_without_callbacks(c) for c in config]
        else:
            _config = _without_callbacks(config)

        response = await self.async_client.post(
            "/batch",
            json={
                "inputs": simple_dumpd(inputs),
                "config": _config,
                "kwargs": kwargs,
            },
        )
        _raise_for_status(response)
        return simple_loads(response.text)["output"]

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[RunnableConfig] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[Output]:
        """Batch invoke the runnable."""
        if kwargs:
            raise NotImplementedError("kwargs not implemented yet.")
        if not inputs:
            return []
        return await self._abatch_with_config(self._abatch, inputs, config)

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """Stream invoke the runnable."""
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)

        final_output: Optional[Output] = None

        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            simple_dumpd(input),
            name=config.get("run_name"),
        )
        data = {
            "input": simple_dumpd(input),
            "config": _without_callbacks(config),
            "kwargs": kwargs,
        }
        endpoint = urljoin(self.url, "stream")

        try:
            from httpx_sse import connect_sse
        except ImportError:
            raise ImportError(
                "Missing `httpx_sse` dependency to use the stream method. "
                "Install via `pip install httpx_sse`'"
            )

        try:
            with connect_sse(
                self.sync_client, "POST", endpoint, json=data
            ) as event_source:
                for sse in event_source.iter_sse():
                    if sse.event == "data":
                        chunk = simple_loads(sse.data)
                        yield chunk

                        if final_output:
                            final_output += chunk
                        else:
                            final_output = chunk
                    elif sse.event == "end":
                        break
                    else:
                        raise NotImplementedError(f"Unknown event {sse.event}")
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(final_output)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)

        final_output: Optional[Output] = None

        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            simple_dumpd(input),
            name=config.get("run_name"),
        )
        data = {
            "input": simple_dumpd(input),
            "config": _without_callbacks(config),
            "kwargs": kwargs,
        }
        endpoint = urljoin(self.url, "stream")

        try:
            from httpx_sse import aconnect_sse
        except ImportError:
            raise ImportError("You must install `httpx_sse` to use the stream method.")

        try:
            async with aconnect_sse(
                self.async_client, "POST", endpoint, json=data
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.event == "data":
                        chunk = simple_loads(sse.data)
                        yield chunk

                        if final_output:
                            final_output += chunk
                        else:
                            final_output = chunk
                    elif sse.event == "end":
                        break

                    else:
                        raise NotImplementedError(f"Unknown event {sse.event}")
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(final_output)

    async def astream_log(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        *,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[RunLogPatch]:
        """Stream all output from a runnable, as reported to the callback system.
        This includes all inner runs of LLMs, Retrievers, Tools, etc.

        Output is streamed as Log objects, which include a list of
        jsonpatch ops that describe how the state of the run has changed in each
        step, and the final state of the run.

        The jsonpatch ops can be applied in order to construct state.
        """

        # Create a stream handler that will emit Log objects
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)

        final_output: Optional[Output] = None

        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            simple_dumpd(input),
            name=config.get("run_name"),
        )
        data = {
            "input": simple_dumpd(input),
            "config": _without_callbacks(config),
            "kwargs": kwargs,
            "include_names": include_names,
            "include_types": include_types,
            "include_tags": include_tags,
            "exclude_names": exclude_names,
            "exclude_types": exclude_types,
            "exclude_tags": exclude_tags,
        }
        endpoint = urljoin(self.url, "stream_log")

        try:
            from httpx_sse import aconnect_sse
        except ImportError:
            raise ImportError("You must install `httpx_sse` to use the stream method.")

        try:
            async with aconnect_sse(
                self.async_client, "POST", endpoint, json=data
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.event == "data":
                        data = simple_loads(sse.data)
                        chunk = RunLogPatch(*data["ops"])
                        yield chunk

                        if final_output:
                            final_output += chunk
                        else:
                            final_output = chunk
                    elif sse.event == "end":
                        break
                    else:
                        raise NotImplementedError(f"Unknown event {sse.event}")
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(final_output)
