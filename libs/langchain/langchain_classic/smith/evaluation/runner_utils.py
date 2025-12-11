"""Utilities for running language models or Chains over datasets."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from langchain_core._api import warn_deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables import config as runnable_config
from langchain_core.runnables import utils as runnable_utils
from langchain_core.tracers.evaluation import (
    EvaluatorCallbackHandler,
    wait_for_all_evaluators,
)
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client
from langsmith.env import get_git_info, get_langchain_env_var_metadata
from langsmith.evaluation import (
    EvaluationResult,
    RunEvaluator,
)
from langsmith.evaluation import (
    run_evaluator as run_evaluator_dec,
)
from langsmith.run_helpers import as_runnable, is_traceable_function
from langsmith.schemas import Dataset, DataType, Example, Run, TracerSession
from langsmith.utils import LangSmithError
from requests import HTTPError
from typing_extensions import TypedDict

from langchain_classic.chains.base import Chain
from langchain_classic.evaluation.loading import load_evaluator
from langchain_classic.evaluation.schema import (
    EvaluatorType,
    PairwiseStringEvaluator,
    StringEvaluator,
)
from langchain_classic.smith import evaluation as smith_eval
from langchain_classic.smith.evaluation import config as smith_eval_config
from langchain_classic.smith.evaluation import name_generation, progress

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

MODEL_OR_CHAIN_FACTORY = (
    Callable[[], Chain | Runnable]
    | BaseLanguageModel
    | Callable[[dict], Any]
    | Runnable
    | Chain
)
MCF = Callable[[], Chain | Runnable] | BaseLanguageModel


class InputFormatError(Exception):
    """Raised when the input format is invalid."""


## Shared Utilities


class TestResult(dict):
    """A dictionary of the results of a single test run."""

    def get_aggregate_feedback(
        self,
    ) -> pd.DataFrame:
        """Return quantiles for the feedback scores.

        This method calculates and prints the quantiles for the feedback scores
        across all feedback keys.

        Returns:
            A DataFrame containing the quantiles for each feedback key.
        """
        df = self.to_dataframe()
        # Drop all things starting with inputs., outputs., and reference
        to_drop = [
            col
            for col in df.columns
            if col.startswith(("inputs.", "outputs.", "reference"))
            or col in {"input", "output"}
        ]
        return df.describe(include="all").drop(to_drop, axis=1)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the results to a dataframe."""
        try:
            import pandas as pd
        except ImportError as e:
            msg = (
                "Pandas is required to convert the results to a dataframe."
                " to install pandas, run `pip install pandas`."
            )
            raise ImportError(msg) from e

        indices = []
        records = []
        for example_id, result in self["results"].items():
            feedback = result["feedback"]
            output_ = result.get("output")
            if isinstance(output_, dict):
                output = {f"outputs.{k}": v for k, v in output_.items()}
            elif output_ is None:
                output = {}
            else:
                output = {"output": output_}

            r = {
                **{f"inputs.{k}": v for k, v in result["input"].items()},
                **output,
            }
            if "reference" in result:
                if isinstance(result["reference"], dict):
                    r.update(
                        {f"reference.{k}": v for k, v in result["reference"].items()},
                    )
                else:
                    r["reference"] = result["reference"]
            r.update(
                {
                    **{f"feedback.{f.key}": f.score for f in feedback},
                    "error": result.get("Error"),
                    "execution_time": result["execution_time"],
                    "run_id": result.get("run_id"),
                },
            )
            records.append(r)
            indices.append(example_id)

        return pd.DataFrame(records, index=indices)


class EvalError(dict):
    """Your architecture raised an error."""

    def __init__(self, Error: BaseException, **kwargs: Any) -> None:  # noqa: N803
        """Initialize the `EvalError` with an error and additional attributes.

        Args:
            Error: The error that occurred.
            **kwargs: Additional attributes to include in the error.
        """
        super().__init__(Error=Error, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the `EvalError`.

        Args:
            name: The name of the attribute to get.

        Returns:
            The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        try:
            return self[name]
        except KeyError as e:
            msg = f"'EvalError' object has no attribute '{name}'"
            raise AttributeError(msg) from e


def _wrap_in_chain_factory(
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    dataset_name: str = "<my_dataset>",
) -> MCF:
    """Wrap in a chain factory.

    Forgive the user if they pass in a chain without memory instead of a chain
    factory. It's a common mistake. Raise a more helpful error message as well.
    """
    if isinstance(llm_or_chain_factory, Chain):
        chain = llm_or_chain_factory
        chain_class = chain.__class__.__name__
        if llm_or_chain_factory.memory is not None:
            memory_class = chain.memory.__class__.__name__
            msg = (
                "Cannot directly evaluate a chain with stateful memory."
                " To evaluate this chain, pass in a chain constructor"
                " that initializes fresh memory each time it is called."
                "  This will safegaurd against information"
                " leakage between dataset examples."
                "\nFor example:\n\n"
                "def chain_constructor():\n"
                f"    new_memory = {memory_class}(...)\n"
                f"    return {chain_class}"
                "(memory=new_memory, ...)\n\n"
                f'run_on_dataset("{dataset_name}", chain_constructor, ...)'
            )
            raise ValueError(msg)
        return lambda: chain
    if isinstance(llm_or_chain_factory, BaseLanguageModel):
        return llm_or_chain_factory
    if isinstance(llm_or_chain_factory, Runnable):
        # Memory may exist here, but it's not elegant to check all those cases.
        lcf = llm_or_chain_factory
        return lambda: lcf
    if callable(llm_or_chain_factory):
        if is_traceable_function(llm_or_chain_factory):
            runnable_ = as_runnable(cast("Callable", llm_or_chain_factory))
            return lambda: runnable_
        try:
            _model = llm_or_chain_factory()  # type: ignore[call-arg]
        except TypeError:
            # It's an arbitrary function, wrap it in a RunnableLambda
            user_func = cast("Callable", llm_or_chain_factory)
            sig = inspect.signature(user_func)
            logger.info("Wrapping function %s as RunnableLambda.", sig)
            wrapped = RunnableLambda(user_func)
            return lambda: wrapped
        constructor = cast("Callable", llm_or_chain_factory)
        if isinstance(_model, BaseLanguageModel):
            # It's not uncommon to do an LLM constructor instead of raw LLM,
            # so we'll unpack it for the user.
            return _model
        if is_traceable_function(cast("Callable", _model)):
            runnable_ = as_runnable(cast("Callable", _model))
            return lambda: runnable_
        if not isinstance(_model, Runnable):
            # This is unlikely to happen - a constructor for a model function
            return lambda: RunnableLambda(constructor)
        # Typical correct case
        return constructor
    return llm_or_chain_factory  # type: ignore[unreachable]


def _get_prompt(inputs: dict[str, Any]) -> str:
    """Get prompt from inputs.

    Args:
        inputs: The input dictionary.

    Returns:
        A string prompt.

    Raises:
        InputFormatError: If the input format is invalid.
    """
    if not inputs:
        msg = "Inputs should not be empty."
        raise InputFormatError(msg)

    prompts = []
    if "prompt" in inputs:
        if not isinstance(inputs["prompt"], str):
            msg = f"Expected string for 'prompt', got {type(inputs['prompt']).__name__}"
            raise InputFormatError(msg)
        prompts = [inputs["prompt"]]
    elif "prompts" in inputs:
        if not isinstance(inputs["prompts"], list) or not all(
            isinstance(i, str) for i in inputs["prompts"]
        ):
            msg = (
                "Expected list of strings for 'prompts',"
                f" got {type(inputs['prompts']).__name__}"
            )
            raise InputFormatError(msg)
        prompts = inputs["prompts"]
    elif len(inputs) == 1:
        prompt_ = next(iter(inputs.values()))
        if isinstance(prompt_, str):
            prompts = [prompt_]
        elif isinstance(prompt_, list) and all(isinstance(i, str) for i in prompt_):
            prompts = prompt_
        else:
            msg = f"LLM Run expects string prompt input. Got {inputs}"
            raise InputFormatError(msg)
    else:
        msg = f"LLM Run expects 'prompt' or 'prompts' in inputs. Got {inputs}"
        raise InputFormatError(msg)
    if len(prompts) == 1:
        return prompts[0]
    msg = f"LLM Run expects single prompt input. Got {len(prompts)} prompts."
    raise InputFormatError(msg)


class ChatModelInput(TypedDict):
    """Input for a chat model."""

    messages: list[BaseMessage]


def _get_messages(inputs: dict[str, Any]) -> dict:
    """Get Chat Messages from inputs.

    Args:
        inputs: The input dictionary.

    Returns:
        A list of chat messages.

    Raises:
        InputFormatError: If the input format is invalid.
    """
    if not inputs:
        msg = "Inputs should not be empty."
        raise InputFormatError(msg)
    input_copy = inputs.copy()
    if "messages" in inputs:
        input_copy["input"] = input_copy.pop("messages")
    elif len(inputs) == 1:
        input_copy["input"] = next(iter(inputs.values()))
    if "input" in input_copy:
        raw_messages = input_copy["input"]
        if isinstance(raw_messages, list) and all(
            isinstance(i, dict) for i in raw_messages
        ):
            raw_messages = [raw_messages]
        if len(raw_messages) == 1:
            input_copy["input"] = messages_from_dict(raw_messages[0])
        else:
            msg = (
                "Batch messages not supported. Please provide a"
                " single list of messages."
            )
            raise InputFormatError(msg)
        return input_copy
    msg = (
        f"Chat Run expects single List[dict] or List[List[dict]] 'messages'"
        f" input. Got {inputs}"
    )
    raise InputFormatError(msg)


## Shared data validation utilities
def _validate_example_inputs_for_language_model(
    first_example: Example,
    input_mapper: Callable[[dict], Any] | None,
) -> None:
    if input_mapper:
        prompt_input = input_mapper(first_example.inputs or {})
        if not isinstance(prompt_input, str) and not (
            isinstance(prompt_input, list)
            and all(isinstance(msg, BaseMessage) for msg in prompt_input)
        ):
            msg = (
                "When using an input_mapper to prepare dataset example inputs"
                " for an LLM or chat model, the output must a single string or"
                " a list of chat messages."
                f"\nGot: {prompt_input} of type {type(prompt_input)}."
            )
            raise InputFormatError(msg)
    else:
        try:
            _get_prompt(first_example.inputs or {})
        except InputFormatError:
            try:
                _get_messages(first_example.inputs or {})
            except InputFormatError as err2:
                msg = (
                    "Example inputs do not match language model input format. "
                    "Expected a dictionary with messages or a single prompt."
                    f" Got: {first_example.inputs}"
                    " Please update your dataset OR provide an input_mapper"
                    " to convert the example.inputs to a compatible format"
                    " for the llm or chat model you wish to evaluate."
                )
                raise InputFormatError(msg) from err2


def _validate_example_inputs_for_chain(
    first_example: Example,
    chain: Chain,
    input_mapper: Callable[[dict], Any] | None,
) -> None:
    """Validate that the example inputs match the chain input keys."""
    if input_mapper:
        first_inputs = input_mapper(first_example.inputs or {})
        missing_keys = set(chain.input_keys).difference(first_inputs)
        if not isinstance(first_inputs, dict):
            msg = (
                "When using an input_mapper to prepare dataset example"
                " inputs for a chain, the mapped value must be a dictionary."
                f"\nGot: {first_inputs} of type {type(first_inputs)}."
            )
            raise InputFormatError(msg)
        if missing_keys:
            msg = (
                "Missing keys after loading example using input_mapper."
                f"\nExpected: {chain.input_keys}. Got: {first_inputs.keys()}"
            )
            raise InputFormatError(msg)
    else:
        first_inputs = first_example.inputs or {}
        missing_keys = set(chain.input_keys).difference(first_inputs)
        if len(first_inputs) == 1 and len(chain.input_keys) == 1:
            # We can pass this through the run method.
            # Refrain from calling to validate.
            pass
        elif missing_keys:
            msg = (
                "Example inputs missing expected chain input keys."
                " Please provide an input_mapper to convert the example.inputs"
                " to a compatible format for the chain you wish to evaluate."
                f"Expected: {chain.input_keys}. "
                f"Got: {first_inputs.keys()}"
            )
            raise InputFormatError(msg)


def _validate_example_inputs(
    example: Example,
    llm_or_chain_factory: MCF,
    input_mapper: Callable[[dict], Any] | None,
) -> None:
    """Validate that the example inputs are valid for the model."""
    if isinstance(llm_or_chain_factory, BaseLanguageModel):
        _validate_example_inputs_for_language_model(example, input_mapper)
    else:
        chain = llm_or_chain_factory()
        if isinstance(chain, Chain):
            # Otherwise it's a runnable
            _validate_example_inputs_for_chain(example, chain, input_mapper)
        elif isinstance(chain, Runnable):
            logger.debug("Skipping input validation for %s", chain)


## Shared Evaluator Setup Utilities


def _setup_evaluation(
    llm_or_chain_factory: MCF,
    examples: list[Example],
    evaluation: smith_eval.RunEvalConfig | None,
    data_type: DataType,
) -> list[RunEvaluator] | None:
    """Configure the evaluators to run on the results of the chain."""
    if evaluation:
        if isinstance(llm_or_chain_factory, BaseLanguageModel):
            run_inputs, run_outputs = None, None
            run_type = "llm"
        else:
            run_type = "chain"
            chain = llm_or_chain_factory()
            run_inputs = chain.input_keys if isinstance(chain, Chain) else None
            run_outputs = chain.output_keys if isinstance(chain, Chain) else None
        run_evaluators = _load_run_evaluators(
            evaluation,
            run_type,
            data_type,
            list(examples[0].outputs) if examples[0].outputs else None,
            run_inputs,
            run_outputs,
        )
    else:
        # TODO: Create a default helpfulness evaluator
        run_evaluators = None
    return run_evaluators


def _determine_input_key(
    config: smith_eval.RunEvalConfig,
    run_inputs: list[str] | None,
) -> str | None:
    input_key = None
    if config.input_key:
        input_key = config.input_key
        if run_inputs and input_key not in run_inputs:
            logger.warning(
                "Input key %s not in chain's specified input keys %s. "
                "Evaluation behavior may be undefined.",
                input_key,
                run_inputs,
            )
    elif run_inputs and len(run_inputs) == 1:
        input_key = run_inputs[0]
    elif run_inputs is not None and len(run_inputs) > 1:
        logger.warning(
            "Chain expects multiple input keys: %s,"
            " Evaluator is likely to fail. Evaluation behavior may be undefined."
            " Specify an input_key in the RunEvalConfig to avoid this warning.",
            run_inputs,
        )

    return input_key


def _determine_prediction_key(
    config: smith_eval.RunEvalConfig,
    run_outputs: list[str] | None,
) -> str | None:
    prediction_key = None
    if config.prediction_key:
        prediction_key = config.prediction_key
        if run_outputs and prediction_key not in run_outputs:
            logger.warning(
                "Prediction key %s not in chain's specified output keys %s. "
                "Evaluation behavior may be undefined.",
                prediction_key,
                run_outputs,
            )
    elif run_outputs and len(run_outputs) == 1:
        prediction_key = run_outputs[0]
    elif run_outputs is not None and len(run_outputs) > 1:
        logger.warning(
            "Chain expects multiple output keys: %s,"
            " Evaluation behavior may be undefined. Specify a prediction_key"
            " in the RunEvalConfig to avoid this warning.",
            run_outputs,
        )
    return prediction_key


def _determine_reference_key(
    config: smith_eval.RunEvalConfig,
    example_outputs: list[str] | None,
) -> str | None:
    if config.reference_key:
        reference_key = config.reference_key
        if example_outputs and reference_key not in example_outputs:
            msg = (
                f"Reference key {reference_key} not in Dataset"
                f" example outputs: {example_outputs}"
            )
            raise ValueError(msg)
    elif example_outputs and len(example_outputs) == 1:
        reference_key = next(iter(example_outputs))
    else:
        reference_key = None
    return reference_key


def _construct_run_evaluator(
    eval_config: smith_eval_config.SINGLE_EVAL_CONFIG_TYPE
    | smith_eval_config.CUSTOM_EVALUATOR_TYPE,
    eval_llm: BaseLanguageModel | None,
    run_type: str,
    data_type: DataType,
    example_outputs: list[str] | None,
    reference_key: str | None,
    input_key: str | None,
    prediction_key: str | None,
) -> RunEvaluator:
    if isinstance(eval_config, RunEvaluator):
        return eval_config
    if isinstance(eval_config, (EvaluatorType, str)):
        if not isinstance(eval_config, EvaluatorType):
            eval_config = EvaluatorType(eval_config)
        evaluator_ = load_evaluator(eval_config, llm=eval_llm)
        eval_type_tag = eval_config.value
    elif isinstance(eval_config, smith_eval_config.EvalConfig):
        kwargs = {"llm": eval_llm, **eval_config.get_kwargs()}
        evaluator_ = load_evaluator(eval_config.evaluator_type, **kwargs)
        eval_type_tag = eval_config.evaluator_type.value
        # Override keys if specified in the config
        if isinstance(eval_config, smith_eval_config.SingleKeyEvalConfig):
            input_key = eval_config.input_key or input_key
            prediction_key = eval_config.prediction_key or prediction_key
            reference_key = eval_config.reference_key or reference_key
    elif callable(eval_config):
        # Assume we can decorate
        return run_evaluator_dec(eval_config)
    else:
        msg = f"Unknown evaluator type: {type(eval_config)}"
        raise ValueError(msg)  # noqa: TRY004

    if isinstance(evaluator_, StringEvaluator):
        if evaluator_.requires_reference and reference_key is None:
            msg = (
                f"Must specify reference_key in smith_eval.RunEvalConfig to use"
                f" evaluator of type {eval_type_tag} with"
                f" dataset with multiple output keys: {example_outputs}."
            )
            raise ValueError(msg)
        run_evaluator = smith_eval.StringRunEvaluatorChain.from_run_and_data_type(
            evaluator_,
            run_type,
            data_type,
            input_key=input_key,
            prediction_key=prediction_key,
            reference_key=reference_key,
            tags=[eval_type_tag],
        )
    elif isinstance(evaluator_, PairwiseStringEvaluator):
        msg = (
            f"Run evaluator for {eval_type_tag} is not implemented."
            " PairwiseStringEvaluators compare the outputs of two different models"
            " rather than the output of a single model."
            " Did you mean to use a StringEvaluator instead?"
            "\nSee: https://python.langchain.com/docs/guides/evaluation/string/"
        )
        raise NotImplementedError(msg)

    else:
        msg = f"Run evaluator for {eval_type_tag} is not implemented"
        raise NotImplementedError(msg)
    return run_evaluator


def _get_keys(
    config: smith_eval.RunEvalConfig,
    run_inputs: list[str] | None,
    run_outputs: list[str] | None,
    example_outputs: list[str] | None,
) -> tuple[str | None, str | None, str | None]:
    input_key = _determine_input_key(config, run_inputs)
    prediction_key = _determine_prediction_key(config, run_outputs)
    reference_key = _determine_reference_key(config, example_outputs)
    return input_key, prediction_key, reference_key


def _load_run_evaluators(
    config: smith_eval.RunEvalConfig,
    run_type: str,
    data_type: DataType,
    example_outputs: list[str] | None,
    run_inputs: list[str] | None,
    run_outputs: list[str] | None,
) -> list[RunEvaluator]:
    """Load run evaluators from a configuration.

    Args:
        config: Configuration for the run evaluators.
        run_type: The type of run.
        data_type: The type of dataset used in the run.
        example_outputs: The example outputs.
        run_inputs: The input keys for the run.
        run_outputs: The output keys for the run.

    Returns:
        A list of run evaluators.
    """
    run_evaluators = []
    input_key, prediction_key, reference_key = None, None, None
    if config.evaluators or (
        config.custom_evaluators
        and any(isinstance(e, StringEvaluator) for e in config.custom_evaluators)
    ):
        input_key, prediction_key, reference_key = _get_keys(
            config,
            run_inputs,
            run_outputs,
            example_outputs,
        )
    for eval_config in config.evaluators:
        run_evaluator = _construct_run_evaluator(
            eval_config,
            config.eval_llm,
            run_type,
            data_type,
            example_outputs,
            reference_key,
            input_key,
            prediction_key,
        )
        run_evaluators.append(run_evaluator)
    custom_evaluators = config.custom_evaluators or []
    for custom_evaluator in custom_evaluators:
        if isinstance(custom_evaluator, RunEvaluator):
            run_evaluators.append(custom_evaluator)
        elif isinstance(custom_evaluator, StringEvaluator):
            run_evaluators.append(
                smith_eval.StringRunEvaluatorChain.from_run_and_data_type(
                    custom_evaluator,
                    run_type,
                    data_type,
                    input_key=input_key,
                    prediction_key=prediction_key,
                    reference_key=reference_key,
                ),
            )
        elif callable(custom_evaluator):
            run_evaluators.append(run_evaluator_dec(custom_evaluator))
        else:
            msg = (  # type: ignore[unreachable]
                f"Unsupported custom evaluator: {custom_evaluator}."
                f" Expected RunEvaluator or StringEvaluator."
            )
            raise ValueError(msg)  # noqa: TRY004

    return run_evaluators


### Async Helpers


async def _arun_llm(
    llm: BaseLanguageModel,
    inputs: dict[str, Any],
    *,
    tags: list[str] | None = None,
    callbacks: Callbacks = None,
    input_mapper: Callable[[dict], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | BaseMessage:
    """Asynchronously run the language model.

    Args:
        llm: The language model to run.
        inputs: The input dictionary.
        tags: Optional tags to add to the run.
        callbacks: Optional callbacks to use during the run.
        input_mapper: Optional function to map inputs to the expected format.
        metadata: Optional metadata to add to the run.

    Returns:
        The LLMResult or ChatResult.

    Raises:
        ValueError: If the LLM type is unsupported.
        InputFormatError: If the input format is invalid.
    """
    if input_mapper is not None:
        prompt_or_messages = input_mapper(inputs)
        if isinstance(prompt_or_messages, str) or (
            isinstance(prompt_or_messages, list)
            and all(isinstance(msg, BaseMessage) for msg in prompt_or_messages)
        ):
            return await llm.ainvoke(
                prompt_or_messages,
                config=RunnableConfig(
                    callbacks=callbacks,
                    tags=tags or [],
                    metadata=metadata or {},
                ),
            )
        msg = (
            "Input mapper returned invalid format"
            f" {prompt_or_messages}"
            "\nExpected a single string or list of chat messages."
        )
        raise InputFormatError(msg)

    try:
        prompt = _get_prompt(inputs)
        llm_output: str | BaseMessage = await llm.ainvoke(
            prompt,
            config=RunnableConfig(
                callbacks=callbacks,
                tags=tags or [],
                metadata=metadata or {},
            ),
        )
    except InputFormatError:
        llm_inputs = _get_messages(inputs)
        llm_output = await llm.ainvoke(
            **llm_inputs,
            config=RunnableConfig(
                callbacks=callbacks,
                tags=tags or [],
                metadata=metadata or {},
            ),
        )
    return llm_output


async def _arun_chain(
    chain: Chain | Runnable,
    inputs: dict[str, Any],
    callbacks: Callbacks,
    *,
    tags: list[str] | None = None,
    input_mapper: Callable[[dict], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict | str:
    """Run a chain asynchronously on inputs."""
    inputs_ = inputs if input_mapper is None else input_mapper(inputs)
    if (
        isinstance(chain, Chain)
        and isinstance(inputs_, dict)
        and len(inputs_) == 1
        and chain.input_keys
    ):
        val = next(iter(inputs_.values()))
        output = await chain.ainvoke(
            val,
            config=RunnableConfig(
                callbacks=callbacks,
                tags=tags or [],
                metadata=metadata or {},
            ),
        )
    else:
        runnable_config = RunnableConfig(
            tags=tags or [],
            callbacks=callbacks,
            metadata=metadata or {},
        )
        output = await chain.ainvoke(inputs_, config=runnable_config)
    return output


async def _arun_llm_or_chain(
    example: Example,
    config: RunnableConfig,
    *,
    llm_or_chain_factory: MCF,
    input_mapper: Callable[[dict], Any] | None = None,
) -> dict | str | LLMResult | ChatResult:
    """Asynchronously run the Chain or language model.

    Args:
        example: The example to run.
        config: The configuration for the run.
        llm_or_chain_factory: The Chain or language model constructor to run.
        input_mapper: Optional function to map the input to the expected format.

    Returns:
        A list of outputs.
    """
    chain_or_llm = (
        "LLM" if isinstance(llm_or_chain_factory, BaseLanguageModel) else "Chain"
    )
    result = None
    try:
        if isinstance(llm_or_chain_factory, BaseLanguageModel):
            output: Any = await _arun_llm(
                llm_or_chain_factory,
                example.inputs or {},
                tags=config["tags"],
                callbacks=config["callbacks"],
                input_mapper=input_mapper,
                metadata=config.get("metadata"),
            )
        else:
            chain = llm_or_chain_factory()
            output = await _arun_chain(
                chain,
                example.inputs or {},
                tags=config["tags"],
                callbacks=config["callbacks"],
                input_mapper=input_mapper,
                metadata=config.get("metadata"),
            )
        result = output
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "%s failed for example %s with inputs %s\n%s",
            chain_or_llm,
            example.id,
            example.inputs,
            e,
        )
        result = EvalError(Error=e)
    return result


## Sync Utilities


def _run_llm(
    llm: BaseLanguageModel,
    inputs: dict[str, Any],
    callbacks: Callbacks,
    *,
    tags: list[str] | None = None,
    input_mapper: Callable[[dict], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | BaseMessage:
    """Run the language model on the example.

    Args:
        llm: The language model to run.
        inputs: The input dictionary.
        callbacks: The callbacks to use during the run.
        tags: Optional tags to add to the run.
        input_mapper: function to map to the inputs dictionary from an Example
        metadata: Optional metadata to add to the run.

    Returns:
        The LLMResult or ChatResult.

    Raises:
        ValueError: If the LLM type is unsupported.
        InputFormatError: If the input format is invalid.
    """
    # Most of this is legacy code; we could probably remove a lot of it.
    if input_mapper is not None:
        prompt_or_messages = input_mapper(inputs)
        if isinstance(prompt_or_messages, str) or (
            isinstance(prompt_or_messages, list)
            and all(isinstance(msg, BaseMessage) for msg in prompt_or_messages)
        ):
            llm_output: str | BaseMessage = llm.invoke(
                prompt_or_messages,
                config=RunnableConfig(
                    callbacks=callbacks,
                    tags=tags or [],
                    metadata=metadata or {},
                ),
            )
        else:
            msg = (
                "Input mapper returned invalid format: "
                f" {prompt_or_messages}"
                "\nExpected a single string or list of chat messages."
            )
            raise InputFormatError(msg)
    else:
        try:
            llm_prompts = _get_prompt(inputs)
            llm_output = llm.invoke(
                llm_prompts,
                config=RunnableConfig(
                    callbacks=callbacks,
                    tags=tags or [],
                    metadata=metadata or {},
                ),
            )
        except InputFormatError:
            llm_inputs = _get_messages(inputs)
            llm_output = llm.invoke(
                **llm_inputs,
                config=RunnableConfig(callbacks=callbacks, metadata=metadata or {}),
            )
    return llm_output


def _run_chain(
    chain: Chain | Runnable,
    inputs: dict[str, Any],
    callbacks: Callbacks,
    *,
    tags: list[str] | None = None,
    input_mapper: Callable[[dict], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict | str:
    """Run a chain on inputs."""
    inputs_ = inputs if input_mapper is None else input_mapper(inputs)
    if (
        isinstance(chain, Chain)
        and isinstance(inputs_, dict)
        and len(inputs_) == 1
        and chain.input_keys
    ):
        val = next(iter(inputs_.values()))
        output = chain.invoke(
            val,
            config=RunnableConfig(
                callbacks=callbacks,
                tags=tags or [],
                metadata=metadata or {},
            ),
        )
    else:
        runnable_config = RunnableConfig(
            tags=tags or [],
            callbacks=callbacks,
            metadata=metadata or {},
        )
        output = chain.invoke(inputs_, config=runnable_config)
    return output


def _run_llm_or_chain(
    example: Example,
    config: RunnableConfig,
    *,
    llm_or_chain_factory: MCF,
    input_mapper: Callable[[dict], Any] | None = None,
) -> dict | str | LLMResult | ChatResult:
    """Run the Chain or language model synchronously.

    Args:
        example: The example to run.
        config: The configuration for the run.
        llm_or_chain_factory: The Chain or language model constructor to run.
        input_mapper: Optional function to map the input to the expected format.

    Returns:
        The outputs of the model or chain.
    """
    chain_or_llm = (
        "LLM" if isinstance(llm_or_chain_factory, BaseLanguageModel) else "Chain"
    )
    result = None
    try:
        if isinstance(llm_or_chain_factory, BaseLanguageModel):
            output: Any = _run_llm(
                llm_or_chain_factory,
                example.inputs or {},
                config["callbacks"],
                tags=config["tags"],
                input_mapper=input_mapper,
                metadata=config.get("metadata"),
            )
        else:
            chain = llm_or_chain_factory()
            output = _run_chain(
                chain,
                example.inputs or {},
                config["callbacks"],
                tags=config["tags"],
                input_mapper=input_mapper,
                metadata=config.get("metadata"),
            )
        result = output
    except Exception as e:  # noqa: BLE001
        error_type = type(e).__name__
        logger.warning(
            "%s failed for example %s with inputs %s\nError Type: %s, Message: %s",
            chain_or_llm,
            example.id,
            example.inputs,
            error_type,
            e,
        )
        result = EvalError(Error=e)
    return result


def _prepare_eval_run(
    client: Client,
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    project_name: str,
    project_metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    dataset_version: str | datetime | None = None,
) -> tuple[MCF, TracerSession, Dataset, list[Example]]:
    wrapped_model = _wrap_in_chain_factory(llm_or_chain_factory, dataset_name)
    dataset = client.read_dataset(dataset_name=dataset_name)

    examples = list(client.list_examples(dataset_id=dataset.id, as_of=dataset_version))
    if not examples:
        msg = f"Dataset {dataset_name} has no example rows."
        raise ValueError(msg)
    modified_at = [ex.modified_at for ex in examples if ex.modified_at]
    # Should always be defined in practice when fetched,
    # but the typing permits None
    max_modified_at = max(modified_at) if modified_at else None
    inferred_version = max_modified_at.isoformat() if max_modified_at else None

    try:
        project_metadata = project_metadata or {}
        git_info = get_git_info()
        if git_info:
            project_metadata = {
                **project_metadata,
                "git": git_info,
            }

        project_metadata["dataset_version"] = inferred_version
        project = client.create_project(
            project_name,
            reference_dataset_id=dataset.id,
            project_extra={"tags": tags} if tags else {},
            metadata=project_metadata,
        )
    except (HTTPError, ValueError, LangSmithError) as e:
        if "already exists " not in str(e):
            raise
        uid = uuid.uuid4()
        example_msg = f"""
run_on_dataset(
    ...
    project_name="{project_name} - {uid}", # Update since {project_name} already exists
)
"""
        msg = (
            f"Test project {project_name} already exists. Please use a different name:"
            f"\n\n{example_msg}"
        )
        raise ValueError(msg) from e
    comparison_url = dataset.url + f"/compare?selectedSessions={project.id}"
    print(  # noqa: T201
        f"View the evaluation results for project '{project_name}'"
        f" at:\n{comparison_url}\n\n"
        f"View all tests for Dataset {dataset_name} at:\n{dataset.url}",
        flush=True,
    )
    return wrapped_model, project, dataset, examples


class _RowResult(TypedDict, total=False):
    """A dictionary of the results for a single example row."""

    feedback: list[EvaluationResult] | None
    execution_time: float | None
    run_id: str | None


@dataclasses.dataclass
class _DatasetRunContainer:
    """A container to help manage the state of a eval run."""

    client: Client
    project: TracerSession
    wrapped_model: MCF
    examples: list[Example]
    configs: list[RunnableConfig]
    batch_evaluators: list[smith_eval_config.BATCH_EVALUATOR_LIKE] | None = None

    def _merge_test_outputs(
        self,
        batch_results: list,
        all_eval_results: dict[str, _RowResult],
    ) -> dict:
        results: dict = {}
        for example, output in zip(self.examples, batch_results, strict=False):
            row_result = all_eval_results.get(str(example.id), {})
            results[str(example.id)] = {
                "input": example.inputs,
                "feedback": row_result.get("feedback", []),
                "execution_time": row_result.get("execution_time"),
                "run_id": row_result.get("run_id"),
            }
            if isinstance(output, EvalError):
                results[str(example.id)]["Error"] = output.Error
            else:
                results[str(example.id)]["output"] = output
            if example.outputs:
                results[str(example.id)]["reference"] = example.outputs
        return results

    def _run_batch_evaluators(self, runs: dict[str, Run]) -> list[dict]:
        evaluators = self.batch_evaluators
        if not evaluators:
            return []
        runs_list = [runs[str(example.id)] for example in self.examples]
        aggregate_feedback = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for evaluator in evaluators:
                try:
                    result = evaluator(runs_list, self.examples)
                    if isinstance(result, EvaluationResult):
                        result = result.dict()
                    aggregate_feedback.append(cast("dict", result))
                    executor.submit(
                        self.client.create_feedback,
                        **result,
                        run_id=None,
                        project_id=self.project.id,
                    )
                except Exception:
                    logger.exception(
                        "Error running batch evaluator %s", repr(evaluator)
                    )
        return aggregate_feedback

    def _collect_metrics(self) -> tuple[dict[str, _RowResult], dict[str, Run]]:
        all_eval_results: dict = {}
        all_runs: dict = {}
        for c in self.configs:
            for callback in cast("list", c["callbacks"]):
                if isinstance(callback, EvaluatorCallbackHandler):
                    eval_results = callback.logged_eval_results
                    for (_, example_id), v in eval_results.items():
                        all_eval_results.setdefault(str(example_id), {}).update(
                            {"feedback": v},
                        )
                elif isinstance(callback, LangChainTracer):
                    run = callback.latest_run
                    execution_time = (
                        (run.end_time - run.start_time).total_seconds()
                        if run and run.end_time
                        else None
                    )
                    run_id = str(run.id) if run else None
                    all_eval_results.setdefault(str(callback.example_id), {}).update(
                        {
                            "execution_time": execution_time,
                            "run_id": run_id,
                            "run": run,
                        },
                    )
                    all_runs[str(callback.example_id)] = run
        return cast("dict[str, _RowResult]", all_eval_results), all_runs

    def _collect_test_results(
        self,
        batch_results: list[dict | str | LLMResult | ChatResult],
    ) -> TestResult:
        logger.info("Waiting for evaluators to complete.")
        wait_for_all_evaluators()
        all_eval_results, all_runs = self._collect_metrics()
        aggregate_feedback = None
        if self.batch_evaluators:
            logger.info("Running session evaluators.")
            aggregate_feedback = self._run_batch_evaluators(all_runs)
        results = self._merge_test_outputs(batch_results, all_eval_results)
        return TestResult(
            project_name=self.project.name,
            results=results,
            aggregate_metrics=aggregate_feedback,
        )

    def finish(
        self,
        batch_results: list,
        verbose: bool = False,  # noqa: FBT001,FBT002
    ) -> TestResult:
        results = self._collect_test_results(batch_results)
        if verbose:
            try:
                agg_feedback = results.get_aggregate_feedback()
                _display_aggregate_results(agg_feedback)
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to print aggregate feedback: %s", e, exc_info=True)
        try:
            # Closing the project permits name changing and metric optimizations
            self.client.update_project(
                self.project.id,
                end_time=datetime.now(timezone.utc),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to close project: %s", e, exc_info=True)
        return results

    @classmethod
    def prepare(
        cls,
        client: Client,
        dataset_name: str,
        llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
        project_name: str | None,
        evaluation: smith_eval.RunEvalConfig | None = None,
        tags: list[str] | None = None,
        input_mapper: Callable[[dict], Any] | None = None,
        concurrency_level: int = 5,
        project_metadata: dict[str, Any] | None = None,
        revision_id: str | None = None,
        dataset_version: datetime | str | None = None,
    ) -> _DatasetRunContainer:
        project_name = project_name or name_generation.random_name()
        if revision_id:
            if not project_metadata:
                project_metadata = {}
            project_metadata.update({"revision_id": revision_id})
        wrapped_model, project, dataset, examples = _prepare_eval_run(
            client,
            dataset_name,
            llm_or_chain_factory,
            project_name,
            project_metadata=project_metadata,
            tags=tags,
            dataset_version=dataset_version,
        )
        tags = tags or []
        for k, v in (project.metadata.get("git") or {}).items():
            tags.append(f"git:{k}={v}")
        run_metadata = {"dataset_version": project.metadata["dataset_version"]}
        if revision_id:
            run_metadata["revision_id"] = revision_id
        wrapped_model = _wrap_in_chain_factory(llm_or_chain_factory)
        run_evaluators = _setup_evaluation(
            wrapped_model,
            examples,
            evaluation,
            dataset.data_type or DataType.kv,
        )
        _validate_example_inputs(examples[0], wrapped_model, input_mapper)
        progress_bar = progress.ProgressBarCallback(len(examples))
        configs = [
            RunnableConfig(
                callbacks=[
                    LangChainTracer(
                        project_name=project.name,
                        client=client,
                        example_id=example.id,
                    ),
                    EvaluatorCallbackHandler(
                        evaluators=run_evaluators or [],
                        client=client,
                        example_id=example.id,
                        max_concurrency=0,
                    ),
                    progress_bar,
                ],
                tags=tags,
                max_concurrency=concurrency_level,
                metadata=run_metadata,
            )
            for example in examples
        ]
        return cls(
            client=client,
            project=project,
            wrapped_model=wrapped_model,
            examples=examples,
            configs=configs,
            batch_evaluators=evaluation.batch_evaluators if evaluation else None,
        )


def _is_jupyter_environment() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        res = get_ipython()  # type: ignore[no-untyped-call]
        return res is not None and "zmqshell" in str(type(res))
    except ImportError:
        return False


def _display_aggregate_results(aggregate_results: pd.DataFrame) -> None:
    if _is_jupyter_environment():
        from IPython.display import HTML, display

        display(HTML("<h3>Experiment Results:</h3>"))  # type: ignore[no-untyped-call]
        display(aggregate_results)  # type: ignore[no-untyped-call]
    else:
        formatted_string = aggregate_results.to_string(
            float_format=lambda x: f"{x:.2f}",
            justify="right",
        )
        print("\n Experiment Results:")  # noqa: T201
        print(formatted_string)  # noqa: T201


_INPUT_MAPPER_DEP_WARNING = (
    "The input_mapper argument is deprecated and "
    "will be removed in a future release. Please add a "
    " RunnableLambda to your chain to map inputs to the expected format"
    " instead. Example:\n"
    "def construct_chain():\n"
    "    my_chain = ...\n"
    "    input_mapper = {'other_key': 'MyOtherInput', 'my_input_key': x}\n"
    "    return input_mapper | my_chain\n"
    "run_on_dataset(..., llm_or_chain_factory=construct_chain)\n"
    "(See https://api.python.langchain.com/en/latest/schema/"
    "langchain.schema.runnable.base.RunnableLambda.html)"
)

## Public API


async def arun_on_dataset(
    client: Client | None,
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    evaluation: smith_eval.RunEvalConfig | None = None,
    dataset_version: datetime | str | None = None,
    concurrency_level: int = 5,
    project_name: str | None = None,
    project_metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    revision_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run on dataset.

    Run the Chain or language model on a dataset and store traces
    to the specified project name.

    For the (usually faster) async version of this function,
    see `arun_on_dataset`.

    Args:
        dataset_name: Name of the dataset to run the chain on.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        evaluation: Configuration for evaluators to run on the
            results of the chain.
        dataset_version: Optional version of the dataset.
        concurrency_level: The number of async tasks to run concurrently.
        project_name: Name of the project to store the traces in.
            Defaults to `{dataset_name}-{chain class name}-{datetime}`.
        project_metadata: Optional metadata to add to the project.
            Useful for storing information the test variant.
            (prompt version, model version, etc.)
        client: LangSmith client to use to access the dataset and to
            log feedback and run traces.
        verbose: Whether to print progress.
        revision_id: Optional revision identifier to assign this test run to
            track the performance of different versions of your system.
        **kwargs: Should not be used, but is provided for backwards compatibility.

    Returns:
        `dict` containing the run's project name and the resulting model outputs.

    Examples:
    ```python
    from langsmith import Client
    from langchain_openai import ChatOpenAI
    from langchain_classic.chains import LLMChain
    from langchain_classic.smith import smith_eval.RunEvalConfig, run_on_dataset

    # Chains may have memory. Passing in a constructor function lets the
    # evaluation framework avoid cross-contamination between runs.
    def construct_chain():
        model = ChatOpenAI(temperature=0)
        chain = LLMChain.from_string(
            model,
            "What's the answer to {your_input_key}"
        )
        return chain

    # Load off-the-shelf evaluators via config or the EvaluatorType (string or enum)
    evaluation_config = smith_eval.RunEvalConfig(
        evaluators=[
            "qa",  # "Correctness" against a reference answer
            "embedding_distance",
            smith_eval.RunEvalConfig.Criteria("helpfulness"),
            smith_eval.RunEvalConfig.Criteria({
                "fifth-grader-score": "Do you have to be smarter than a fifth "
                "grader to answer this question?"
            }),
        ]
    )

    client = Client()
    await arun_on_dataset(
        client,
        dataset_name="<my_dataset_name>",
        llm_or_chain_factory=construct_chain,
        evaluation=evaluation_config,
    )
    ```
    You can also create custom evaluators by subclassing the `StringEvaluator or
    LangSmith's `RunEvaluator` classes.

    ```python
    from typing import Optional
    from langchain_classic.evaluation import StringEvaluator


    class MyStringEvaluator(StringEvaluator):
        @property
        def requires_input(self) -> bool:
            return False

        @property
        def requires_reference(self) -> bool:
            return True

        @property
        def evaluation_name(self) -> str:
            return "exact_match"

        def _evaluate_strings(
            self, prediction, reference=None, input=None, **kwargs
        ) -> dict:
            return {"score": prediction == reference}


    evaluation_config = smith_eval.RunEvalConfig(
        custom_evaluators=[MyStringEvaluator()],
    )

    await arun_on_dataset(
        client,
        dataset_name="<my_dataset_name>",
        llm_or_chain_factory=construct_chain,
        evaluation=evaluation_config,
    )
    ```
    """
    input_mapper = kwargs.pop("input_mapper", None)
    if input_mapper:
        warn_deprecated("0.0.305", message=_INPUT_MAPPER_DEP_WARNING, pending=True)
    if revision_id is None:
        revision_id = get_langchain_env_var_metadata().get("revision_id")
    tags = kwargs.pop("tags", None)
    if tags:
        warn_deprecated(
            "0.1.9",
            message="The tags argument is deprecated and will be"
            " removed in a future release. Please specify project_metadata instead.",
            pending=True,
        )

    if kwargs:
        warn_deprecated(
            "0.0.305",
            message="The following arguments are deprecated and "
            "will be removed in a future release: "
            f"{kwargs.keys()}.",
            removal="0.0.305",
        )
    client = client or Client()
    container = _DatasetRunContainer.prepare(
        client,
        dataset_name,
        llm_or_chain_factory,
        project_name,
        evaluation,
        tags,
        input_mapper,
        concurrency_level,
        project_metadata=project_metadata,
        revision_id=revision_id,
        dataset_version=dataset_version,
    )
    batch_results = await runnable_utils.gather_with_concurrency(
        container.configs[0].get("max_concurrency"),
        *map(
            functools.partial(
                _arun_llm_or_chain,
                llm_or_chain_factory=container.wrapped_model,
                input_mapper=input_mapper,
            ),
            container.examples,
            container.configs,
        ),
    )
    return container.finish(batch_results, verbose=verbose)


def run_on_dataset(
    client: Client | None,
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    evaluation: smith_eval.RunEvalConfig | None = None,
    dataset_version: datetime | str | None = None,
    concurrency_level: int = 5,
    project_name: str | None = None,
    project_metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    revision_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run on dataset.

    Run the Chain or language model on a dataset and store traces
    to the specified project name.

    For the (usually faster) async version of this function,
    see `arun_on_dataset`.

    Args:
        dataset_name: Name of the dataset to run the chain on.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        evaluation: Configuration for evaluators to run on the
            results of the chain.
        dataset_version: Optional version of the dataset.
        concurrency_level: The number of async tasks to run concurrently.
        project_name: Name of the project to store the traces in.
            Defaults to `{dataset_name}-{chain class name}-{datetime}`.
        project_metadata: Optional metadata to add to the project.
            Useful for storing information the test variant.
            (prompt version, model version, etc.)
        client: LangSmith client to use to access the dataset and to
            log feedback and run traces.
        verbose: Whether to print progress.
        revision_id: Optional revision identifier to assign this test run to
            track the performance of different versions of your system.
        **kwargs: Should not be used, but is provided for backwards compatibility.

    Returns:
        `dict` containing the run's project name and the resulting model outputs.

    Examples:
    ```python
    from langsmith import Client
    from langchain_openai import ChatOpenAI
    from langchain_classic.chains import LLMChain
    from langchain_classic.smith import smith_eval.RunEvalConfig, run_on_dataset

    # Chains may have memory. Passing in a constructor function lets the
    # evaluation framework avoid cross-contamination between runs.
    def construct_chain():
        model = ChatOpenAI(temperature=0)
        chain = LLMChain.from_string(
            model,
            "What's the answer to {your_input_key}"
        )
        return chain

    # Load off-the-shelf evaluators via config or the EvaluatorType (string or enum)
    evaluation_config = smith_eval.RunEvalConfig(
        evaluators=[
            "qa",  # "Correctness" against a reference answer
            "embedding_distance",
            smith_eval.RunEvalConfig.Criteria("helpfulness"),
            smith_eval.RunEvalConfig.Criteria({
                "fifth-grader-score": "Do you have to be smarter than a fifth "
                "grader to answer this question?"
            }),
        ]
    )

    client = Client()
    run_on_dataset(
        client,
        dataset_name="<my_dataset_name>",
        llm_or_chain_factory=construct_chain,
        evaluation=evaluation_config,
    )
    ```

    You can also create custom evaluators by subclassing the `StringEvaluator` or
    LangSmith's `RunEvaluator` classes.

    ```python
    from typing import Optional
    from langchain_classic.evaluation import StringEvaluator


    class MyStringEvaluator(StringEvaluator):
        @property
        def requires_input(self) -> bool:
            return False

        @property
        def requires_reference(self) -> bool:
            return True

        @property
        def evaluation_name(self) -> str:
            return "exact_match"

        def _evaluate_strings(
            self, prediction, reference=None, input=None, **kwargs
        ) -> dict:
            return {"score": prediction == reference}


    evaluation_config = smith_eval.RunEvalConfig(
        custom_evaluators=[MyStringEvaluator()],
    )

    run_on_dataset(
        client,
        dataset_name="<my_dataset_name>",
        llm_or_chain_factory=construct_chain,
        evaluation=evaluation_config,
    )
    ```
    """
    input_mapper = kwargs.pop("input_mapper", None)
    if input_mapper:
        warn_deprecated("0.0.305", message=_INPUT_MAPPER_DEP_WARNING, pending=True)
    tags = kwargs.pop("tags", None)
    if tags:
        warn_deprecated(
            "0.1.9",
            message="The tags argument is deprecated and will be"
            " removed in a future release. Please specify project_metadata instead.",
            pending=True,
        )
    if revision_id is None:
        revision_id = get_langchain_env_var_metadata().get("revision_id")

    if kwargs:
        warn_deprecated(
            "0.0.305",
            message="The following arguments are deprecated and "
            "will be removed in a future release: "
            f"{kwargs.keys()}.",
            removal="0.0.305",
        )
    client = client or Client()
    container = _DatasetRunContainer.prepare(
        client,
        dataset_name,
        llm_or_chain_factory,
        project_name,
        evaluation,
        tags,
        input_mapper,
        concurrency_level,
        project_metadata=project_metadata,
        revision_id=revision_id,
        dataset_version=dataset_version,
    )
    if concurrency_level == 0:
        batch_results = [
            _run_llm_or_chain(
                example,
                config,
                llm_or_chain_factory=container.wrapped_model,
                input_mapper=input_mapper,
            )
            for example, config in zip(
                container.examples, container.configs, strict=False
            )
        ]
    else:
        with runnable_config.get_executor_for_config(container.configs[0]) as executor:
            batch_results = list(
                executor.map(
                    functools.partial(
                        _run_llm_or_chain,
                        llm_or_chain_factory=container.wrapped_model,
                        input_mapper=input_mapper,
                    ),
                    container.examples,
                    container.configs,
                ),
            )

    return container.finish(batch_results, verbose=verbose)
