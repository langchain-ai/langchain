from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_experimental.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_experimental.rl_chain.metrics import (
    MetricsTrackerAverage,
    MetricsTrackerRollingWindow,
)
from langchain_experimental.rl_chain.model_repository import ModelRepository
from langchain_experimental.rl_chain.vw_logger import VwLogger

if TYPE_CHECKING:
    import vowpal_wabbit_next as vw

logger = logging.getLogger(__name__)


class _BasedOn:
    def __init__(self, value: Any):
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    __repr__ = __str__


def BasedOn(anything: Any) -> _BasedOn:
    """Wrap a value to indicate that it should be based on."""

    return _BasedOn(anything)


class _ToSelectFrom:
    def __init__(self, value: Any):
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    __repr__ = __str__


def ToSelectFrom(anything: Any) -> _ToSelectFrom:
    """Wrap a value to indicate that it should be selected from."""

    if not isinstance(anything, list):
        raise ValueError("ToSelectFrom must be a list to select from")
    return _ToSelectFrom(anything)


class _Embed:
    def __init__(self, value: Any, keep: bool = False):
        self.value = value
        self.keep = keep

    def __str__(self) -> str:
        return str(self.value)

    __repr__ = __str__


def Embed(anything: Any, keep: bool = False) -> Any:
    """Wrap a value to indicate that it should be embedded."""

    if isinstance(anything, _ToSelectFrom):
        return ToSelectFrom(Embed(anything.value, keep=keep))
    elif isinstance(anything, _BasedOn):
        return BasedOn(Embed(anything.value, keep=keep))
    if isinstance(anything, list):
        return [Embed(v, keep=keep) for v in anything]
    elif isinstance(anything, dict):
        return {k: Embed(v, keep=keep) for k, v in anything.items()}
    elif isinstance(anything, _Embed):
        return anything
    return _Embed(anything, keep=keep)


def EmbedAndKeep(anything: Any) -> Any:
    """Wrap a value to indicate that it should be embedded and kept."""

    return Embed(anything, keep=True)


# helper functions


def stringify_embedding(embedding: List) -> str:
    """Convert an embedding to a string."""

    return " ".join([f"{i}:{e}" for i, e in enumerate(embedding)])


def parse_lines(parser: "vw.TextFormatParser", input_str: str) -> List["vw.Example"]:
    """Parse the input string into a list of examples."""

    return [parser.parse_line(line) for line in input_str.split("\n")]


def get_based_on_and_to_select_from(inputs: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Get the BasedOn and ToSelectFrom from the inputs."""
    to_select_from = {
        k: inputs[k].value
        for k in inputs.keys()
        if isinstance(inputs[k], _ToSelectFrom)
    }

    if not to_select_from:
        raise ValueError(
            "No variables using 'ToSelectFrom' found in the inputs. Please include at least one variable containing a list to select from."  # noqa: E501
        )

    based_on = {
        k: inputs[k].value if isinstance(inputs[k].value, list) else [inputs[k].value]
        for k in inputs.keys()
        if isinstance(inputs[k], _BasedOn)
    }

    return based_on, to_select_from


def prepare_inputs_for_autoembed(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the inputs for auto embedding.

    Go over all the inputs and if something is either wrapped in _ToSelectFrom or _BasedOn, and if their inner values are not already _Embed,
    then wrap them in EmbedAndKeep while retaining their _ToSelectFrom or _BasedOn status
    """  # noqa: E501

    next_inputs = inputs.copy()
    for k, v in next_inputs.items():
        if isinstance(v, _ToSelectFrom) or isinstance(v, _BasedOn):
            if not isinstance(v.value, _Embed):
                next_inputs[k].value = EmbedAndKeep(v.value)
    return next_inputs


# end helper functions


class Selected(ABC):
    """Abstract class to represent the selected item."""

    pass


TSelected = TypeVar("TSelected", bound=Selected)


class Event(Generic[TSelected], ABC):
    """Abstract class to represent an event."""

    inputs: Dict[str, Any]
    selected: Optional[TSelected]

    def __init__(self, inputs: Dict[str, Any], selected: Optional[TSelected] = None):
        self.inputs = inputs
        self.selected = selected


TEvent = TypeVar("TEvent", bound=Event)


class Policy(Generic[TEvent], ABC):
    """Abstract class to represent a policy."""

    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
    def predict(self, event: TEvent) -> Any:
        ...

    @abstractmethod
    def learn(self, event: TEvent) -> None:
        ...

    @abstractmethod
    def log(self, event: TEvent) -> None:
        ...

    def save(self) -> None:
        pass


class VwPolicy(Policy):
    """Vowpal Wabbit policy."""

    def __init__(
        self,
        model_repo: ModelRepository,
        vw_cmd: List[str],
        feature_embedder: Embedder,
        vw_logger: VwLogger,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.model_repo = model_repo
        self.workspace = self.model_repo.load(vw_cmd)
        self.feature_embedder = feature_embedder
        self.vw_logger = vw_logger

    def predict(self, event: TEvent) -> Any:
        import vowpal_wabbit_next as vw

        text_parser = vw.TextFormatParser(self.workspace)
        return self.workspace.predict_one(
            parse_lines(text_parser, self.feature_embedder.format(event))
        )

    def learn(self, event: TEvent) -> None:
        import vowpal_wabbit_next as vw

        vw_ex = self.feature_embedder.format(event)
        text_parser = vw.TextFormatParser(self.workspace)
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)

    def log(self, event: TEvent) -> None:
        if self.vw_logger.logging_enabled():
            vw_ex = self.feature_embedder.format(event)
            self.vw_logger.log(vw_ex)

    def save(self) -> None:
        self.model_repo.save(self.workspace)


class Embedder(Generic[TEvent], ABC):
    """Abstract class to represent an embedder."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def format(self, event: TEvent) -> str:
        ...


class SelectionScorer(Generic[TEvent], ABC, BaseModel):
    """Abstract class to grade the chosen selection or the response of the llm."""

    @abstractmethod
    def score_response(
        self, inputs: Dict[str, Any], llm_response: str, event: TEvent
    ) -> float:
        ...


class AutoSelectionScorer(SelectionScorer[Event], BaseModel):
    """Auto selection scorer."""

    llm_chain: LLMChain
    prompt: Union[BasePromptTemplate, None] = None
    scoring_criteria_template_str: Optional[str] = None

    @staticmethod
    def get_default_system_prompt() -> SystemMessagePromptTemplate:
        return SystemMessagePromptTemplate.from_template(
            "PLEASE RESPOND ONLY WITH A SINGLE FLOAT AND NO OTHER TEXT EXPLANATION\n \
                You are a strict judge that is called on to rank a response based on \
                    given criteria. You must respond with your ranking by providing a \
                        single float within the range [0, 1], 0 being very bad \
                            response and 1 being very good response."
        )

    @staticmethod
    def get_default_prompt() -> ChatPromptTemplate:
        human_template = 'Given this based_on "{rl_chain_selected_based_on}" \
            as the most important attribute, rank how good or bad this text is: \
                "{rl_chain_selected}".'
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        default_system_prompt = AutoSelectionScorer.get_default_system_prompt()
        chat_prompt = ChatPromptTemplate.from_messages(
            [default_system_prompt, human_message_prompt]
        )
        return chat_prompt

    @root_validator(pre=True)
    def set_prompt_and_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        llm = values.get("llm")
        prompt = values.get("prompt")
        scoring_criteria_template_str = values.get("scoring_criteria_template_str")
        if prompt is None and scoring_criteria_template_str is None:
            prompt = AutoSelectionScorer.get_default_prompt()
        elif prompt is None and scoring_criteria_template_str is not None:
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                scoring_criteria_template_str
            )
            default_system_prompt = AutoSelectionScorer.get_default_system_prompt()
            prompt = ChatPromptTemplate.from_messages(
                [default_system_prompt, human_message_prompt]
            )
        values["prompt"] = prompt
        values["llm_chain"] = LLMChain(llm=llm, prompt=prompt)
        return values

    def score_response(
        self, inputs: Dict[str, Any], llm_response: str, event: Event
    ) -> float:
        ranking = self.llm_chain.predict(llm_response=llm_response, **inputs)
        ranking = ranking.strip()
        try:
            resp = float(ranking)
            return resp
        except Exception as e:
            raise RuntimeError(
                f"The auto selection scorer did not manage to score the response, there is always the option to try again or tweak the reward prompt. Error: {e}"  # noqa: E501
            )


class RLChain(Chain, Generic[TEvent]):
    """Chain that leverages the Vowpal Wabbit (VW) model as a learned policy
    for reinforcement learning.

    Attributes:
        - llm_chain (Chain): Represents the underlying Language Model chain.
        - prompt (BasePromptTemplate): The template for the base prompt.
        - selection_scorer (Union[SelectionScorer, None]): Scorer for the selection. Can be set to None.
        - policy (Optional[Policy]): The policy used by the chain to learn to populate a dynamic prompt.
        - auto_embed (bool): Determines if embedding should be automatic. Default is False.
        - metrics (Optional[Union[MetricsTrackerRollingWindow, MetricsTrackerAverage]]): Tracker for metrics, can be set to None.

    Initialization Attributes:
        - feature_embedder (Embedder): Embedder used for the `BasedOn` and `ToSelectFrom` inputs.
        - model_save_dir (str, optional): Directory for saving the VW model. Default is the current directory.
        - reset_model (bool): If set to True, the model starts training from scratch. Default is False.
        - vw_cmd (List[str], optional): Command line arguments for the VW model.
        - policy (Type[VwPolicy]): Policy used by the chain.
        - vw_logs (Optional[Union[str, os.PathLike]]): Path for the VW logs.
        - metrics_step (int): Step for the metrics tracker. Default is -1. If set without metrics_window_size, average metrics will be tracked, otherwise rolling window metrics will be tracked.
        - metrics_window_size (int): Window size for the metrics tracker. Default is -1. If set, rolling window metrics will be tracked.

    Notes:
        The class initializes the VW model using the provided arguments. If `selection_scorer` is not provided, a warning is logged, indicating that no reinforcement learning will occur unless the `update_with_delayed_score` method is called.
    """  # noqa: E501

    class _NoOpPolicy(Policy):
        """Placeholder policy that does nothing"""

        def predict(self, event: TEvent) -> Any:
            return None

        def learn(self, event: TEvent) -> None:
            pass

        def log(self, event: TEvent) -> None:
            pass

    llm_chain: Chain

    output_key: str = "result"  #: :meta private:
    prompt: BasePromptTemplate
    selection_scorer: Union[SelectionScorer, None]
    active_policy: Policy = _NoOpPolicy()
    auto_embed: bool = False
    selection_scorer_activated: bool = True
    selected_input_key = "rl_chain_selected"
    selected_based_on_input_key = "rl_chain_selected_based_on"
    metrics: Optional[Union[MetricsTrackerRollingWindow, MetricsTrackerAverage]] = None

    def __init__(
        self,
        feature_embedder: Embedder,
        model_save_dir: str = "./",
        reset_model: bool = False,
        vw_cmd: Optional[List[str]] = None,
        policy: Type[Policy] = VwPolicy,
        vw_logs: Optional[Union[str, os.PathLike]] = None,
        metrics_step: int = -1,
        metrics_window_size: int = -1,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if self.selection_scorer is None:
            logger.warning(
                "No selection scorer provided, which means that no \
                    reinforcement learning will be done in the RL chain \
                        unless update_with_delayed_score is called."
            )

        if isinstance(self.active_policy, RLChain._NoOpPolicy):
            self.active_policy = policy(
                model_repo=ModelRepository(
                    model_save_dir, with_history=True, reset=reset_model
                ),
                vw_cmd=vw_cmd or [],
                feature_embedder=feature_embedder,
                vw_logger=VwLogger(vw_logs),
            )

        if metrics_window_size > 0:
            self.metrics = MetricsTrackerRollingWindow(
                step=metrics_step, window_size=metrics_window_size
            )
        else:
            self.metrics = MetricsTrackerAverage(step=metrics_step)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return []

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def update_with_delayed_score(
        self, score: float, chain_response: Dict[str, Any], force_score: bool = False
    ) -> None:
        """
        Updates the learned policy with the score provided.
        Will raise an error if selection_scorer is set, and force_score=True was not provided during the method call
        """  # noqa: E501
        if self._can_use_selection_scorer() and not force_score:
            raise RuntimeError(
                "The selection scorer is set, and force_score was not set to True. Please set force_score=True to use this function."  # noqa: E501
            )
        if self.metrics:
            self.metrics.on_feedback(score)
        event: TEvent = chain_response["selection_metadata"]
        self._call_after_scoring_before_learning(event=event, score=score)
        self.active_policy.learn(event=event)
        self.active_policy.log(event=event)

    def deactivate_selection_scorer(self) -> None:
        """
        Deactivates the selection scorer, meaning that the chain will no longer attempt to use the selection scorer to score responses.
        """  # noqa: E501
        self.selection_scorer_activated = False

    def activate_selection_scorer(self) -> None:
        """
        Activates the selection scorer, meaning that the chain will attempt to use the selection scorer to score responses.
        """  # noqa: E501
        self.selection_scorer_activated = True

    def save_progress(self) -> None:
        """
        This function should be called to save the state of the learned policy model.
        """  # noqa: E501
        self.active_policy.save()

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        super()._validate_inputs(inputs)
        if (
            self.selected_input_key in inputs.keys()
            or self.selected_based_on_input_key in inputs.keys()
        ):
            raise ValueError(
                f"The rl chain does not accept '{self.selected_input_key}' or '{self.selected_based_on_input_key}' as input keys, they are reserved for internal use during auto reward."  # noqa: E501
            )

    def _can_use_selection_scorer(self) -> bool:
        """
        Returns whether the chain can use the selection scorer to score responses or not.
        """  # noqa: E501
        return self.selection_scorer is not None and self.selection_scorer_activated

    @abstractmethod
    def _call_before_predict(self, inputs: Dict[str, Any]) -> TEvent:
        ...

    @abstractmethod
    def _call_after_predict_before_llm(
        self, inputs: Dict[str, Any], event: TEvent, prediction: Any
    ) -> Tuple[Dict[str, Any], TEvent]:
        ...

    @abstractmethod
    def _call_after_llm_before_scoring(
        self, llm_response: str, event: TEvent
    ) -> Tuple[Dict[str, Any], TEvent]:
        ...

    @abstractmethod
    def _call_after_scoring_before_learning(
        self, event: TEvent, score: Optional[float]
    ) -> TEvent:
        ...

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        event: TEvent = self._call_before_predict(inputs=inputs)
        prediction = self.active_policy.predict(event=event)
        if self.metrics:
            self.metrics.on_decision()

        next_chain_inputs, event = self._call_after_predict_before_llm(
            inputs=inputs, event=event, prediction=prediction
        )

        t = self.llm_chain.run(**next_chain_inputs, callbacks=_run_manager.get_child())
        _run_manager.on_text(t, color="green", verbose=self.verbose)
        t = t.strip()

        if self.verbose:
            _run_manager.on_text("\nCode: ", verbose=self.verbose)

        output = t
        _run_manager.on_text("\nAnswer: ", verbose=self.verbose)
        _run_manager.on_text(output, color="yellow", verbose=self.verbose)

        next_chain_inputs, event = self._call_after_llm_before_scoring(
            llm_response=output, event=event
        )

        score = None
        try:
            if self._can_use_selection_scorer():
                score = self.selection_scorer.score_response(  # type: ignore
                    inputs=next_chain_inputs, llm_response=output, event=event
                )
        except Exception as e:
            logger.info(
                f"The selection scorer was not able to score, \
                and the chain was not able to adjust to this response, error: {e}"
            )
        if self.metrics and score is not None:
            self.metrics.on_feedback(score)

        event = self._call_after_scoring_before_learning(score=score, event=event)
        self.active_policy.learn(event=event)
        self.active_policy.log(event=event)

        return {self.output_key: {"response": output, "selection_metadata": event}}

    @property
    def _chain_type(self) -> str:
        return "llm_personalizer_chain"


def is_stringtype_instance(item: Any) -> bool:
    """Check if an item is a string."""

    return isinstance(item, str) or (
        isinstance(item, _Embed) and isinstance(item.value, str)
    )


def embed_string_type(
    item: Union[str, _Embed], model: Any, namespace: Optional[str] = None
) -> Dict[str, Union[str, List[str]]]:
    """Embed a string or an _Embed object."""

    keep_str = ""
    if isinstance(item, _Embed):
        encoded = stringify_embedding(model.encode(item.value))
        if item.keep:
            keep_str = item.value.replace(" ", "_") + " "
    elif isinstance(item, str):
        encoded = item.replace(" ", "_")
    else:
        raise ValueError(f"Unsupported type {type(item)} for embedding")

    if namespace is None:
        raise ValueError(
            "The default namespace must be provided when embedding a string or _Embed object."  # noqa: E501
        )

    return {namespace: keep_str + encoded}


def embed_dict_type(item: Dict, model: Any) -> Dict[str, Any]:
    """Embed a dictionary item."""
    inner_dict: Dict = {}
    for ns, embed_item in item.items():
        if isinstance(embed_item, list):
            inner_dict[ns] = []
            for embed_list_item in embed_item:
                embedded = embed_string_type(embed_list_item, model, ns)
                inner_dict[ns].append(embedded[ns])
        else:
            inner_dict.update(embed_string_type(embed_item, model, ns))
    return inner_dict


def embed_list_type(
    item: list, model: Any, namespace: Optional[str] = None
) -> List[Dict[str, Union[str, List[str]]]]:
    """Embed a list item."""

    ret_list: List = []
    for embed_item in item:
        if isinstance(embed_item, dict):
            ret_list.append(embed_dict_type(embed_item, model))
        elif isinstance(embed_item, list):
            item_embedding = embed_list_type(embed_item, model, namespace)
            # Get the first key from the first dictionary
            first_key = next(iter(item_embedding[0]))
            # Group the values under that key
            grouping = {first_key: [item[first_key] for item in item_embedding]}
            ret_list.append(grouping)
        else:
            ret_list.append(embed_string_type(embed_item, model, namespace))
    return ret_list


def embed(
    to_embed: Union[Union[str, _Embed], Dict, List[Union[str, _Embed]], List[Dict]],
    model: Any,
    namespace: Optional[str] = None,
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Embed the actions or context using the SentenceTransformer model
    (or a model that has an `encode` function).

    Attributes:
        to_embed: (Union[Union(str, _Embed(str)), Dict, List[Union(str, _Embed(str))], List[Dict]], required) The text to be embedded, either a string, a list of strings or a dictionary or a list of dictionaries.
        namespace: (str, optional) The default namespace to use when dictionary or list of dictionaries not provided.
        model: (Any, required) The model to use for embedding
    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary has the namespace as the key and the embedded string as the value
    """  # noqa: E501
    if (isinstance(to_embed, _Embed) and isinstance(to_embed.value, str)) or isinstance(
        to_embed, str
    ):
        return [embed_string_type(to_embed, model, namespace)]
    elif isinstance(to_embed, dict):
        return [embed_dict_type(to_embed, model)]
    elif isinstance(to_embed, list):
        return embed_list_type(to_embed, model, namespace)
    else:
        raise ValueError("Invalid input format for embedding")
