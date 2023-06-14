"""Utilities for running LLMs/Chains over datasets."""
from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, Iterator, List, Optional, Union

from langchainplus_sdk import LangChainPlusClient
from langchainplus_sdk.schemas import Example

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import Callbacks
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.schema import (
    BaseMessage,
    ChatResult,
    HumanMessage,
    LLMResult,
    get_buffer_string,
    messages_from_dict,
)

logger = logging.getLogger(__name__)

MODEL_OR_CHAIN_FACTORY = Union[Callable[[], Chain], BaseLanguageModel]


class InputFormatError(Exception):
    """Raised when input format is invalid."""


def _get_prompts(inputs: Dict[str, Any]) -> List[str]:
    """Get prompts from inputs."""
    if not inputs:
        raise InputFormatError("Inputs should not be empty.")

    prompts = []
    if "prompt" in inputs:
        if not isinstance(inputs["prompt"], str):
            raise InputFormatError(
                "Expected string for 'prompt', got"
                f" {type(inputs['prompt']).__name__}"
            )
        prompts = [inputs["prompt"]]
    elif "prompts" in inputs:
        if not isinstance(inputs["prompts"], list) or not all(
            isinstance(i, str) for i in inputs["prompts"]
        ):
            raise InputFormatError(
                "Expected list of strings for 'prompts',"
                f" got {type(inputs['prompts']).__name__}"
            )
        prompts = inputs["prompts"]
    elif len(inputs) == 1:
        prompt_ = next(iter(inputs.values()))
        if isinstance(prompt_, str):
            prompts = [prompt_]
        elif isinstance(prompt_, list) and all(isinstance(i, str) for i in prompt_):
            prompts = prompt_
        else:
            raise InputFormatError(f"LLM Run expects string prompt input. Got {inputs}")
    else:
        raise InputFormatError(
            f"LLM Run expects 'prompt' or 'prompts' in inputs. Got {inputs}"
        )

    return prompts


def _get_messages(inputs: Dict[str, Any]) -> List[List[BaseMessage]]:
    """Get Chat Messages from inputs."""
    if not inputs:
        raise InputFormatError("Inputs should not be empty.")

    if "messages" in inputs:
        single_input = inputs["messages"]
    elif len(inputs) == 1:
        single_input = next(iter(inputs.values()))
    else:
        raise InputFormatError(f"Chat Run expects 'messages' in inputs. Got {inputs}")
    if isinstance(single_input, list) and all(
        isinstance(i, dict) for i in single_input
    ):
        raw_messages = [single_input]
    elif isinstance(single_input, list) and all(
        isinstance(i, list) for i in single_input
    ):
        raw_messages = single_input
    else:
        raise InputFormatError(
            f"Chat Run expects List[dict] or List[List[dict]] 'messages'"
            f" input. Got {inputs}"
        )
    return [messages_from_dict(batch) for batch in raw_messages]


async def _arun_llm(
    llm: BaseLanguageModel,
    inputs: Dict[str, Any],
    langchain_tracer: Optional[LangChainTracer],
) -> Union[LLMResult, ChatResult]:
    callbacks: Optional[List[BaseCallbackHandler]] = (
        [langchain_tracer] if langchain_tracer else None
    )
    if isinstance(llm, BaseLLM):
        try:
            llm_prompts = _get_prompts(inputs)
            llm_output = await llm.agenerate(llm_prompts, callbacks=callbacks)
        except InputFormatError:
            llm_messages = _get_messages(inputs)
            buffer_strings = [get_buffer_string(messages) for messages in llm_messages]
            llm_output = await llm.agenerate(buffer_strings, callbacks=callbacks)
    elif isinstance(llm, BaseChatModel):
        try:
            messages = _get_messages(inputs)
            llm_output = await llm.agenerate(messages, callbacks=callbacks)
        except InputFormatError:
            prompts = _get_prompts(inputs)
            converted_messages: List[List[BaseMessage]] = [
                [HumanMessage(content=prompt)] for prompt in prompts
            ]
            llm_output = await llm.agenerate(converted_messages, callbacks=callbacks)
    else:
        raise ValueError(f"Unsupported LLM type {type(llm)}")
    return llm_output


async def _arun_llm_or_chain(
    example: Example,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    n_repetitions: int,
    langchain_tracer: Optional[LangChainTracer],
) -> Union[List[dict], List[str], List[LLMResult], List[ChatResult]]:
    """Run the chain asynchronously."""
    if langchain_tracer is not None:
        previous_example_id = langchain_tracer.example_id
        langchain_tracer.example_id = example.id
        callbacks: Optional[List[BaseCallbackHandler]] = [langchain_tracer]
    else:
        previous_example_id = None
        callbacks = None
    outputs = []
    for _ in range(n_repetitions):
        try:
            if isinstance(llm_or_chain_factory, BaseLanguageModel):
                output: Any = await _arun_llm(
                    llm_or_chain_factory, example.inputs, langchain_tracer
                )
            else:
                chain = llm_or_chain_factory()
                output = await chain.acall(example.inputs, callbacks=callbacks)
            outputs.append(output)
        except Exception as e:
            logger.warning(f"Chain failed for example {example.id}. Error: {e}")
            outputs.append({"Error": str(e)})
    if langchain_tracer is not None:
        langchain_tracer.example_id = previous_example_id
    return outputs


async def _gather_with_concurrency(
    n: int,
    initializer: Callable[[], Coroutine[Any, Any, Optional[LangChainTracer]]],
    *async_funcs: Callable[[Optional[LangChainTracer], Dict], Coroutine[Any, Any, Any]],
) -> List[Any]:
    """
    Run coroutines with a concurrency limit.

    Args:
        n: The maximum number of concurrent tasks.
        initializer: A coroutine that initializes shared resources for the tasks.
        async_funcs: The async_funcs to be run concurrently.

    Returns:
        A list of results from the coroutines.
    """
    semaphore = asyncio.Semaphore(n)
    job_state = {"num_processed": 0}

    tracer_queue: asyncio.Queue[Optional[LangChainTracer]] = asyncio.Queue()
    for _ in range(n):
        tracer_queue.put_nowait(await initializer())

    async def run_coroutine_with_semaphore(
        async_func: Callable[
            [Optional[LangChainTracer], Dict], Coroutine[Any, Any, Any]
        ]
    ) -> Any:
        async with semaphore:
            tracer = await tracer_queue.get()
            try:
                result = await async_func(tracer, job_state)
            finally:
                tracer_queue.put_nowait(tracer)
            return result

    return await asyncio.gather(
        *(run_coroutine_with_semaphore(function) for function in async_funcs)
    )


async def _tracer_initializer(session_name: Optional[str]) -> Optional[LangChainTracer]:
    """
    Initialize a tracer to share across tasks.

    Args:
        session_name: The session name for the tracer.

    Returns:
        A LangChainTracer instance with an active session.
    """
    if session_name:
        tracer = LangChainTracer(session_name=session_name)
        return tracer
    else:
        return None


async def arun_on_examples(
    examples: Iterator[Example],
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    concurrency_level: int = 5,
    num_repetitions: int = 1,
    session_name: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the chain on examples and store traces to the specified session name.

    Args:
        examples: Examples to run the model or chain over
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        concurrency_level: The number of async tasks to run concurrently.
        num_repetitions: Number of times to run the model on each example.
            This is useful when testing success rates or generating confidence
            intervals.
        session_name: Session name to use when tracing runs.
        verbose: Whether to print progress.

    Returns:
        A dictionary mapping example ids to the model outputs.
    """
    results: Dict[str, List[Any]] = {}

    async def process_example(
        example: Example, tracer: LangChainTracer, job_state: dict
    ) -> None:
        """Process a single example."""
        result = await _arun_llm_or_chain(
            example,
            llm_or_chain_factory,
            num_repetitions,
            tracer,
        )
        results[str(example.id)] = result
        job_state["num_processed"] += 1
        if verbose:
            print(
                f"Processed examples: {job_state['num_processed']}",
                end="\r",
                flush=True,
            )

    await _gather_with_concurrency(
        concurrency_level,
        functools.partial(_tracer_initializer, session_name),
        *(functools.partial(process_example, e) for e in examples),
    )
    return results


def run_llm(
    llm: BaseLanguageModel,
    inputs: Dict[str, Any],
    callbacks: Callbacks,
) -> Union[LLMResult, ChatResult]:
    """Run the language model on the example."""
    if isinstance(llm, BaseLLM):
        try:
            llm_prompts = _get_prompts(inputs)
            llm_output = llm.generate(llm_prompts, callbacks=callbacks)
        except InputFormatError:
            llm_messages = _get_messages(inputs)
            buffer_strings = [get_buffer_string(messages) for messages in llm_messages]
            llm_output = llm.generate(buffer_strings, callbacks=callbacks)
    elif isinstance(llm, BaseChatModel):
        try:
            messages = _get_messages(inputs)
            llm_output = llm.generate(messages, callbacks=callbacks)
        except InputFormatError:
            prompts = _get_prompts(inputs)
            converted_messages: List[List[BaseMessage]] = [
                [HumanMessage(content=prompt)] for prompt in prompts
            ]
            llm_output = llm.generate(converted_messages, callbacks=callbacks)
    else:
        raise ValueError(f"Unsupported LLM type {type(llm)}")
    return llm_output


def run_llm_or_chain(
    example: Example,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    n_repetitions: int,
    langchain_tracer: Optional[LangChainTracer] = None,
) -> Union[List[dict], List[str], List[LLMResult], List[ChatResult]]:
    """Run the chain synchronously."""
    if langchain_tracer is not None:
        previous_example_id = langchain_tracer.example_id
        langchain_tracer.example_id = example.id
        callbacks: Optional[List[BaseCallbackHandler]] = [langchain_tracer]
    else:
        previous_example_id = None
        callbacks = None
    outputs = []
    for _ in range(n_repetitions):
        try:
            if isinstance(llm_or_chain_factory, BaseLanguageModel):
                output: Any = run_llm(llm_or_chain_factory, example.inputs, callbacks)
            else:
                chain = llm_or_chain_factory()
                output = chain(example.inputs, callbacks=callbacks)
            outputs.append(output)
        except Exception as e:
            logger.warning(f"Chain failed for example {example.id}. Error: {e}")
            outputs.append({"Error": str(e)})
    if langchain_tracer is not None:
        langchain_tracer.example_id = previous_example_id
    return outputs


def run_on_examples(
    examples: Iterator[Example],
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    num_repetitions: int = 1,
    session_name: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the chain on examples and store traces to the specified session name.

    Args:
        examples: Examples to run model or chain over.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        concurrency_level: Number of async workers to run in parallel.
        num_repetitions: Number of times to run the model on each example.
            This is useful when testing success rates or generating confidence
            intervals.
        session_name: Session name to use when tracing runs.
        verbose: Whether to print progress.
    Returns:
        A dictionary mapping example ids to the model outputs.
    """
    results: Dict[str, Any] = {}
    tracer = LangChainTracer(session_name=session_name) if session_name else None
    for i, example in enumerate(examples):
        result = run_llm_or_chain(
            example,
            llm_or_chain_factory,
            num_repetitions,
            langchain_tracer=tracer,
        )
        if verbose:
            print(f"{i+1} processed", flush=True, end="\r")
    results[str(example.id)] = result
    return results


def _get_session_name(
    session_name: Optional[str],
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    dataset_name: str,
) -> str:
    if session_name is not None:
        return session_name
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if isinstance(llm_or_chain_factory, BaseLanguageModel):
        model_name = llm_or_chain_factory.__class__.__name__
    else:
        model_name = llm_or_chain_factory().__class__.__name__
    return f"{dataset_name}-{model_name}-{current_time}"


async def arun_on_dataset(
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    concurrency_level: int = 5,
    num_repetitions: int = 1,
    session_name: Optional[str] = None,
    verbose: bool = False,
    client: Optional[LangChainPlusClient] = None,
) -> Dict[str, Any]:
    """
    Run the chain on a dataset and store traces to the specified session name.

    Args:
        client: Client to use to read the dataset.
        dataset_name: Name of the dataset to run the chain on.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        concurrency_level: The number of async tasks to run concurrently.
        num_repetitions: Number of times to run the model on each example.
            This is useful when testing success rates or generating confidence
            intervals.
        session_name: Name of the session to store the traces in.
            Defaults to {dataset_name}-{chain class name}-{datetime}.
        verbose: Whether to print progress.
        client: Client to use to read the dataset. If not provided, a new
            client will be created using the credentials in the environment.

    Returns:
        A dictionary containing the run's session name and the resulting model outputs.
    """
    client_ = client or LangChainPlusClient()
    session_name = _get_session_name(session_name, llm_or_chain_factory, dataset_name)
    dataset = client_.read_dataset(dataset_name=dataset_name)
    examples = client_.list_examples(dataset_id=str(dataset.id))

    results = await arun_on_examples(
        examples,
        llm_or_chain_factory,
        concurrency_level=concurrency_level,
        num_repetitions=num_repetitions,
        session_name=session_name,
        verbose=verbose,
    )
    return {
        "session_name": session_name,
        "results": results,
    }


def run_on_dataset(
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    num_repetitions: int = 1,
    session_name: Optional[str] = None,
    verbose: bool = False,
    client: Optional[LangChainPlusClient] = None,
) -> Dict[str, Any]:
    """Run the chain on a dataset and store traces to the specified session name.

    Args:
        dataset_name: Name of the dataset to run the chain on.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        concurrency_level: Number of async workers to run in parallel.
        num_repetitions: Number of times to run the model on each example.
            This is useful when testing success rates or generating confidence
            intervals.
        session_name: Name of the session to store the traces in.
            Defaults to {dataset_name}-{chain class name}-{datetime}.
        verbose: Whether to print progress.
        client: Client to use to access the dataset. If None, a new client
            will be created using the credentials in the environment.

    Returns:
        A dictionary containing the run's session name and the resulting model outputs.
    """
    client_ = client or LangChainPlusClient()
    session_name = _get_session_name(session_name, llm_or_chain_factory, dataset_name)
    dataset = client_.read_dataset(dataset_name=dataset_name)
    examples = client_.list_examples(dataset_id=str(dataset.id))
    results = run_on_examples(
        examples,
        llm_or_chain_factory,
        num_repetitions=num_repetitions,
        session_name=session_name,
        verbose=verbose,
    )
    return {
        "session_name": session_name,
        "results": results,
    }
