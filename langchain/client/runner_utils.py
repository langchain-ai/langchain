"""Utilities for running language models or Chains over datasets."""

from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

from langchainplus_sdk import LangChainPlusClient, RunEvaluator
from langchainplus_sdk.schemas import Example

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import Callbacks
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.evaluation import EvaluatorCallbackHandler
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
    """Raised when the input format is invalid."""


def _get_prompts(inputs: Dict[str, Any]) -> List[str]:
    """
    Get prompts from inputs.

    Args:
        inputs: The input dictionary.

    Returns:
        A list of prompts.
    Raises:
        InputFormatError: If the input format is invalid.
    """
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
    """
    Get Chat Messages from inputs.

    Args:
        inputs: The input dictionary.

    Returns:
        A list of chat messages.
    Raises:
        InputFormatError: If the input format is invalid.
    """
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
    *,
    tags: Optional[List[str]] = None,
    callbacks: Callbacks = None,
    input_mapper: Optional[Callable[[Dict], Any]] = None,
) -> Union[LLMResult, ChatResult]:
    """
    Asynchronously run the language model.

    Args:
        llm: The language model to run.
        inputs: The input dictionary.
        tags: Optional tags to add to the run.
        callbacks: Optional callbacks to use during the run.
        input_mapper: Optional function to map inputs to the expected format.

    Returns:
        The LLMResult or ChatResult.
    Raises:
        ValueError: If the LLM type is unsupported.
        InputFormatError: If the input format is invalid.
    """
    if input_mapper is not None:
        if not isinstance(llm, (BaseLLM, BaseChatModel)):
            raise ValueError(f"Unsupported LLM type {type(llm).__name__}")
        llm_output = await llm.agenerate(
            input_mapper(inputs), callbacks=callbacks, tags=tags
        )
    elif isinstance(llm, BaseLLM):
        try:
            llm_prompts = _get_prompts(inputs)
            llm_output = await llm.agenerate(
                llm_prompts, callbacks=callbacks, tags=tags
            )
        except InputFormatError:
            llm_messages = _get_messages(inputs)
            buffer_strings = [get_buffer_string(messages) for messages in llm_messages]
            llm_output = await llm.agenerate(
                buffer_strings, callbacks=callbacks, tags=tags
            )
    elif isinstance(llm, BaseChatModel):
        try:
            messages = _get_messages(inputs)
            llm_output = await llm.agenerate(messages, callbacks=callbacks, tags=tags)
        except InputFormatError:
            prompts = _get_prompts(inputs)
            converted_messages: List[List[BaseMessage]] = [
                [HumanMessage(content=prompt)] for prompt in prompts
            ]
            llm_output = await llm.agenerate(
                converted_messages, callbacks=callbacks, tags=tags
            )
    else:
        raise ValueError(f"Unsupported LLM type {type(llm)}")
    return llm_output


async def _arun_llm_or_chain(
    example: Example,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    n_repetitions: int,
    *,
    tags: Optional[List[str]] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    input_mapper: Optional[Callable[[Dict], Any]] = None,
) -> Union[List[dict], List[str], List[LLMResult], List[ChatResult]]:
    """
    Asynchronously run the Chain or language model.

    Args:
        example: The example to run.
        llm_or_chain_factory: The Chain or language model constructor to run.
        n_repetitions: The number of times to run the model on each example.
        tags: Optional tags to add to the run.
        callbacks: Optional callbacks to use during the run.
        input_mapper: Optional function to map the input to the expected format.

    Returns:
        A list of outputs.
    """
    if callbacks:
        previous_example_ids = [
            getattr(tracer, "example_id", None) for tracer in callbacks
        ]
        for tracer in callbacks:
            if hasattr(tracer, "example_id"):
                tracer.example_id = example.id
    else:
        previous_example_ids = None
    outputs = []
    for _ in range(n_repetitions):
        try:
            if isinstance(llm_or_chain_factory, BaseLanguageModel):
                output: Any = await _arun_llm(
                    llm_or_chain_factory,
                    example.inputs,
                    tags=tags,
                    callbacks=callbacks,
                    input_mapper=input_mapper,
                )
            else:
                chain = llm_or_chain_factory()
                if input_mapper is not None:
                    inputs_ = input_mapper(example.inputs)
                else:
                    inputs_ = example.inputs
                    if len(inputs_) == 1:
                        inputs_ = next(iter(inputs_.values()))
                output = await chain.acall(inputs_, callbacks=callbacks, tags=tags)
            outputs.append(output)
        except Exception as e:
            logger.warning(f"Chain failed for example {example.id}. Error: {e}")
            outputs.append({"Error": str(e)})
    if callbacks and previous_example_ids:
        for example_id, tracer in zip(previous_example_ids, callbacks):
            if hasattr(tracer, "example_id"):
                tracer.example_id = example_id
    return outputs


async def _gather_with_concurrency(
    n: int,
    initializer: Callable[[], Coroutine[Any, Any, Any]],
    *async_funcs: Callable[
        [Sequence[BaseCallbackHandler], Dict], Coroutine[Any, Any, Any]
    ],
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

    callback_queue: asyncio.Queue[Sequence[BaseCallbackHandler]] = asyncio.Queue()
    for _ in range(n):
        callback_queue.put_nowait(await initializer())

    async def run_coroutine_with_semaphore(
        async_func: Callable[
            [Sequence[BaseCallbackHandler], Dict], Coroutine[Any, Any, Any]
        ]
    ) -> Any:
        async with semaphore:
            callbacks = await callback_queue.get()
            try:
                result = await async_func(callbacks, job_state)
            finally:
                callback_queue.put_nowait(callbacks)
            return result

    results = await asyncio.gather(
        *(run_coroutine_with_semaphore(function) for function in async_funcs)
    )
    while callback_queue:
        try:
            callbacks = callback_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        for callback in callbacks:
            if isinstance(callback, (LangChainTracer, EvaluatorCallbackHandler)):
                callback.wait_for_futures()
    return results


async def _callbacks_initializer(
    project_name: Optional[str],
    client: LangChainPlusClient,
    run_evaluators: Sequence[RunEvaluator],
) -> List[BaseTracer]:
    """
    Initialize a tracer to share across tasks.

    Args:
        project_name: The project name for the tracer.

    Returns:
        A LangChainTracer instance with an active project.
    """
    callbacks: List[BaseTracer] = []
    if project_name:
        callbacks.append(LangChainTracer(project_name=project_name))
    if run_evaluators:
        callbacks.append(
            EvaluatorCallbackHandler(
                client=client,
                evaluators=run_evaluators,
                # We already have concurrency, don't want to overload the machine
                max_workers=1,
            )
        )
    return callbacks


async def arun_on_examples(
    examples: Iterator[Example],
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    concurrency_level: int = 5,
    num_repetitions: int = 1,
    project_name: Optional[str] = None,
    verbose: bool = False,
    client: Optional[LangChainPlusClient] = None,
    tags: Optional[List[str]] = None,
    run_evaluators: Optional[Sequence[RunEvaluator]] = None,
    input_mapper: Optional[Callable[[Dict], Any]] = None,
) -> Dict[str, Any]:
    """
    Asynchronously run the chain on examples and store traces
        to the specified project name.

    Args:
        examples: Examples to run the model or chain over.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        concurrency_level: The number of async tasks to run concurrently.
        num_repetitions: Number of times to run the model on each example.
            This is useful when testing success rates or generating confidence
            intervals.
        project_name: Project name to use when tracing runs.
            Defaults to {dataset_name}-{chain class name}-{datetime}.
        verbose: Whether to print progress.
        client: Client to use to read the dataset. If not provided, a new
            client will be created using the credentials in the environment.
        tags: Tags to add to each run in the project.
        run_evaluators: Evaluators to run on the results of the chain.
        input_mapper: function to map to the inputs dictionary from an Example
            to the format expected by the model to be evaluated. This is useful if
            your model needs to deserialize more complex schema or if your dataset
            has inputs with keys that differ from what is expected by your chain
            or agent.

    Returns:
        A dictionary mapping example ids to the model outputs.
    """
    project_name = _get_project_name(project_name, llm_or_chain_factory, None)
    client_ = client or LangChainPlusClient()
    client_.create_project(project_name, mode="eval")

    results: Dict[str, List[Any]] = {}
    evaluation_handler = EvaluatorCallbackHandler(
        evaluators=run_evaluators or [], client=client_
    )

    async def process_example(
        example: Example, callbacks: List[BaseCallbackHandler], job_state: dict
    ) -> None:
        """Process a single example."""
        result = await _arun_llm_or_chain(
            example,
            llm_or_chain_factory,
            num_repetitions,
            tags=tags,
            callbacks=callbacks,
            input_mapper=input_mapper,
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
        functools.partial(
            _callbacks_initializer,
            project_name=project_name,
            client=client_,
            run_evaluators=run_evaluators or [],
        ),
        *(functools.partial(process_example, e) for e in examples),
    )
    evaluation_handler.wait_for_futures()
    return results


def run_llm(
    llm: BaseLanguageModel,
    inputs: Dict[str, Any],
    callbacks: Callbacks,
    *,
    tags: Optional[List[str]] = None,
    input_mapper: Optional[Callable[[Dict], Any]] = None,
) -> Union[LLMResult, ChatResult]:
    """
    Run the language model on the example.

    Args:
        llm: The language model to run.
        inputs: The input dictionary.
        callbacks: The callbacks to use during the run.
        tags: Optional tags to add to the run.
        input_mapper: function to map to the inputs dictionary from an Example
    Returns:
        The LLMResult or ChatResult.
    Raises:
        ValueError: If the LLM type is unsupported.
        InputFormatError: If the input format is invalid.
    """
    if input_mapper is not None:
        if not isinstance(llm, (BaseLLM, BaseChatModel)):
            raise ValueError(f"Unsupported LLM type {type(llm).__name__}")
        llm_output = llm.generate(input_mapper(inputs), callbacks=callbacks, tags=tags)
    elif isinstance(llm, BaseLLM):
        try:
            llm_prompts = _get_prompts(inputs)
            llm_output = llm.generate(llm_prompts, callbacks=callbacks, tags=tags)
        except InputFormatError:
            llm_messages = _get_messages(inputs)
            buffer_strings = [get_buffer_string(messages) for messages in llm_messages]
            llm_output = llm.generate(buffer_strings, callbacks=callbacks)
    elif isinstance(llm, BaseChatModel):
        try:
            messages = _get_messages(inputs)
            llm_output = llm.generate(messages, callbacks=callbacks, tags=tags)
        except InputFormatError:
            prompts = _get_prompts(inputs)
            converted_messages: List[List[BaseMessage]] = [
                [HumanMessage(content=prompt)] for prompt in prompts
            ]
            llm_output = llm.generate(
                converted_messages, callbacks=callbacks, tags=tags
            )
    else:
        raise ValueError(f"Unsupported LLM type {type(llm)}")
    return llm_output


def run_llm_or_chain(
    example: Example,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    n_repetitions: int,
    *,
    tags: Optional[List[str]] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    input_mapper: Optional[Callable[[Dict], Any]] = None,
) -> Union[List[dict], List[str], List[LLMResult], List[ChatResult]]:
    """
    Run the Chain or language model synchronously.

    Args:
        example: The example to run.
        llm_or_chain_factory: The Chain or language model constructor to run.
        n_repetitions: The number of times to run the model on each example.
        tags: Optional tags to add to the run.
        callbacks: Optional callbacks to use during the run.

    Returns:
        A list of outputs.
    """
    if callbacks:
        previous_example_ids = [
            getattr(tracer, "example_id", None) for tracer in callbacks
        ]
        for tracer in callbacks:
            if hasattr(tracer, "example_id"):
                tracer.example_id = example.id
    else:
        previous_example_ids = None
    outputs = []
    for _ in range(n_repetitions):
        try:
            if isinstance(llm_or_chain_factory, BaseLanguageModel):
                output: Any = run_llm(
                    llm_or_chain_factory,
                    example.inputs,
                    callbacks,
                    tags=tags,
                    input_mapper=input_mapper,
                )
            else:
                chain = llm_or_chain_factory()
                if input_mapper is not None:
                    inputs_ = input_mapper(example.inputs)
                else:
                    inputs_ = example.inputs
                    if len(inputs_) == 1:
                        inputs_ = next(iter(inputs_.values()))
                output = chain(inputs_, callbacks=callbacks, tags=tags)
            outputs.append(output)
        except Exception as e:
            logger.warning(f"Chain failed for example {example.id}. Error: {e}")
            outputs.append({"Error": str(e)})
    if callbacks and previous_example_ids:
        for example_id, tracer in zip(previous_example_ids, callbacks):
            if hasattr(tracer, "example_id"):
                tracer.example_id = example_id
    return outputs


def run_on_examples(
    examples: Iterator[Example],
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    num_repetitions: int = 1,
    project_name: Optional[str] = None,
    verbose: bool = False,
    client: Optional[LangChainPlusClient] = None,
    tags: Optional[List[str]] = None,
    run_evaluators: Optional[Sequence[RunEvaluator]] = None,
    input_mapper: Optional[Callable[[Dict], Any]] = None,
) -> Dict[str, Any]:
    """
    Run the Chain or language model on examples and store
    traces to the specified project name.

    Args:
        examples: Examples to run the model or chain over.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        num_repetitions: Number of times to run the model on each example.
            This is useful when testing success rates or generating confidence
            intervals.
        project_name: Name of the project to store the traces in.
            Defaults to {dataset_name}-{chain class name}-{datetime}.
        verbose: Whether to print progress.
        client: Client to use to access the dataset. If None, a new client
            will be created using the credentials in the environment.
        tags: Tags to add to each run in the project.
        run_evaluators: Evaluators to run on the results of the chain.
        input_mapper: A function to map to the inputs dictionary from an Example
            to the format expected by the model to be evaluated. This is useful if
            your model needs to deserialize more complex schema or if your dataset
            has inputs with keys that differ from what is expected by your chain
            or agent.

    Returns:
        A dictionary mapping example ids to the model outputs.
    """
    results: Dict[str, Any] = {}
    project_name = _get_project_name(project_name, llm_or_chain_factory, None)
    client_ = client or LangChainPlusClient()
    client_.create_project(project_name, mode="eval")
    tracer = LangChainTracer(project_name=project_name)
    evalution_handler = EvaluatorCallbackHandler(
        evaluators=run_evaluators or [], client=client_
    )
    callbacks: List[BaseCallbackHandler] = [tracer, evalution_handler]
    for i, example in enumerate(examples):
        result = run_llm_or_chain(
            example,
            llm_or_chain_factory,
            num_repetitions,
            tags=tags,
            callbacks=callbacks,
            input_mapper=input_mapper,
        )
        if verbose:
            print(f"{i+1} processed", flush=True, end="\r")
        results[str(example.id)] = result
    tracer.wait_for_futures()
    evalution_handler.wait_for_futures()
    return results


def _get_project_name(
    project_name: Optional[str],
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    dataset_name: Optional[str],
) -> str:
    """
    Get the project name.

    Args:
        project_name: The project name if manually specified.
        llm_or_chain_factory: The Chain or language model constructor.
        dataset_name: The dataset name.

    Returns:
        The project name.
    """
    if project_name is not None:
        return project_name
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if isinstance(llm_or_chain_factory, BaseLanguageModel):
        model_name = llm_or_chain_factory.__class__.__name__
    else:
        model_name = llm_or_chain_factory().__class__.__name__
    dataset_prefix = f"{dataset_name}-" if dataset_name else ""
    return f"{dataset_prefix}{model_name}-{current_time}"


async def arun_on_dataset(
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    concurrency_level: int = 5,
    num_repetitions: int = 1,
    project_name: Optional[str] = None,
    verbose: bool = False,
    client: Optional[LangChainPlusClient] = None,
    tags: Optional[List[str]] = None,
    run_evaluators: Optional[Sequence[RunEvaluator]] = None,
    input_mapper: Optional[Callable[[Dict], Any]] = None,
) -> Dict[str, Any]:
    """
    Asynchronously run the Chain or language model on a dataset
    and store traces to the specified project name.

    Args:
        dataset_name: Name of the dataset to run the chain on.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        concurrency_level: The number of async tasks to run concurrently.
        num_repetitions: Number of times to run the model on each example.
            This is useful when testing success rates or generating confidence
            intervals.
        project_name: Name of the project to store the traces in.
            Defaults to {dataset_name}-{chain class name}-{datetime}.
        verbose: Whether to print progress.
        client: Client to use to read the dataset. If not provided, a new
            client will be created using the credentials in the environment.
        tags: Tags to add to each run in the project.
        run_evaluators: Evaluators to run on the results of the chain.
        input_mapper: A function to map to the inputs dictionary from an Example
            to the format expected by the model to be evaluated. This is useful if
            your model needs to deserialize more complex schema or if your dataset
            has inputs with keys that differ from what is expected by your chain
            or agent.
    Returns:
        A dictionary containing the run's project name and the resulting model outputs.
    """
    client_ = client or LangChainPlusClient()
    project_name = _get_project_name(project_name, llm_or_chain_factory, dataset_name)
    dataset = client_.read_dataset(dataset_name=dataset_name)
    examples = client_.list_examples(dataset_id=str(dataset.id))
    results = await arun_on_examples(
        examples,
        llm_or_chain_factory,
        concurrency_level=concurrency_level,
        num_repetitions=num_repetitions,
        project_name=project_name,
        verbose=verbose,
        client=client_,
        tags=tags,
        run_evaluators=run_evaluators,
        input_mapper=input_mapper,
    )
    return {
        "project_name": project_name,
        "results": results,
    }


def run_on_dataset(
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    num_repetitions: int = 1,
    project_name: Optional[str] = None,
    verbose: bool = False,
    client: Optional[LangChainPlusClient] = None,
    tags: Optional[List[str]] = None,
    run_evaluators: Optional[Sequence[RunEvaluator]] = None,
    input_mapper: Optional[Callable[[Dict], Any]] = None,
) -> Dict[str, Any]:
    """
    Run the Chain or language model on a dataset and store traces
    to the specified project name.

    Args:
        dataset_name: Name of the dataset to run the chain on.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        concurrency_level: Number of async workers to run in parallel.
        num_repetitions: Number of times to run the model on each example.
            This is useful when testing success rates or generating confidence
            intervals.
        project_name: Name of the project to store the traces in.
            Defaults to {dataset_name}-{chain class name}-{datetime}.
        verbose: Whether to print progress.
        client: Client to use to access the dataset. If None, a new client
            will be created using the credentials in the environment.
        tags: Tags to add to each run in the project.
        run_evaluators: Evaluators to run on the results of the chain.
        input_mapper: A function to map to the inputs dictionary from an Example
            to the format expected by the model to be evaluated. This is useful if
            your model needs to deserialize more complex schema or if your dataset
            has inputs with keys that differ from what is expected by your chain
            or agent.

    Returns:
        A dictionary containing the run's project name and the resulting model outputs.
    """
    client_ = client or LangChainPlusClient()
    project_name = _get_project_name(project_name, llm_or_chain_factory, dataset_name)
    dataset = client_.read_dataset(dataset_name=dataset_name)
    examples = client_.list_examples(dataset_id=str(dataset.id))
    results = run_on_examples(
        examples,
        llm_or_chain_factory,
        num_repetitions=num_repetitions,
        project_name=project_name,
        verbose=verbose,
        tags=tags,
        run_evaluators=run_evaluators,
        client=client_,
        input_mapper=input_mapper,
    )
    return {
        "project_name": project_name,
        "results": results,
    }
