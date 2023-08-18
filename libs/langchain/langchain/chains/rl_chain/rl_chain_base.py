from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from abc import ABC, abstractmethod

import vowpal_wabbit_next as vw
from .vw_logger import VwLogger
from .model_repository import ModelRepository
from .metrics import MetricsTracker
from langchain.prompts import BasePromptTemplate

from langchain.pydantic_v1 import Extra, BaseModel, root_validator

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


class _BasedOn:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


def BasedOn(anything):
    return _BasedOn(anything)


class _ToSelectFrom:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


def ToSelectFrom(anything):
    if not isinstance(anything, list):
        raise ValueError("ToSelectFrom must be a list to select from")
    return _ToSelectFrom(anything)


class _Embed:
    def __init__(self, value, keep=False):
        self.value = value
        self.keep = keep

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


def Embed(anything, keep=False):
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


def EmbedAndKeep(anything):
    return Embed(anything, keep=True)


# helper functions


def parse_lines(parser: vw.TextFormatParser, input_str: str) -> List[vw.Example]:
    return [parser.parse_line(line) for line in input_str.split("\n")]


def get_based_on_and_to_select_from(inputs: Dict[str, Any]):
    to_select_from = {
        k: inputs[k].value
        for k in inputs.keys()
        if isinstance(inputs[k], _ToSelectFrom)
    }

    if not to_select_from:
        raise ValueError(
            "No variables using 'ToSelectFrom' found in the inputs. Please include at least one variable containing a list to select from."
        )

    based_on = {
        k: inputs[k].value if isinstance(inputs[k].value, list) else [inputs[k].value]
        for k in inputs.keys()
        if isinstance(inputs[k], _BasedOn)
    }

    return based_on, to_select_from


def prepare_inputs_for_autoembed(inputs: Dict[str, Any]):
    # go over all the inputs and if something is either wrapped in _ToSelectFrom or _BasedOn, and if
    # their inner values are not already _Embed, then wrap them in EmbedAndKeep while retaining their _ToSelectFrom or _BasedOn status
    next_inputs = inputs.copy()
    for k, v in next_inputs.items():
        if isinstance(v, _ToSelectFrom) or isinstance(v, _BasedOn):
            if not isinstance(v.value, _Embed):
                next_inputs[k].value = EmbedAndKeep(v.value)
    return next_inputs


# end helper functions


class Selected(ABC):
    pass


class Event(ABC):
    inputs: Dict[str, Any]
    selected: Optional[Selected]

    def __init__(self, inputs: Dict[str, Any], selected: Optional[Selected] = None):
        self.inputs = inputs
        self.selected = selected


class Policy(ABC):
    @abstractmethod
    def predict(self, event: Event) -> Any:
        pass

    @abstractmethod
    def learn(self, event: Event):
        pass

    @abstractmethod
    def log(self, event: Event):
        pass

    def save(self):
        pass


class VwPolicy(Policy):
    def __init__(
        self,
        model_repo: ModelRepository,
        vw_cmd: Sequence[str],
        feature_embedder: Embedder,
        vw_logger: VwLogger,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_repo = model_repo
        self.workspace = self.model_repo.load(vw_cmd)
        self.feature_embedder = feature_embedder
        self.vw_logger = vw_logger

    def predict(self, event: Event) -> Any:
        text_parser = vw.TextFormatParser(self.workspace)
        return self.workspace.predict_one(
            parse_lines(text_parser, self.feature_embedder.format(event))
        )

    def learn(self, event: Event):
        vw_ex = self.feature_embedder.format(event)

        text_parser = vw.TextFormatParser(self.workspace)
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)

    def log(self, event: Event):
        if self.vw_logger.logging_enabled():
            vw_ex = self.feature_embedder.format(event)
            self.vw_logger.log(vw_ex)

    def save(self):
        self.model_repo.save()


class Embedder(ABC):
    @abstractmethod
    def format(self, event: Event) -> str:
        pass


class SelectionScorer(ABC, BaseModel):
    """Abstract method to grade the chosen selection or the response of the llm"""

    @abstractmethod
    def score_response(self, inputs: Dict[str, Any], llm_response: str) -> float:
        pass


class AutoSelectionScorer(SelectionScorer, BaseModel):
    llm_chain: Union[LLMChain, None] = None
    prompt: Union[BasePromptTemplate, None] = None
    scoring_criteria_template_str: Optional[str] = None

    @staticmethod
    def get_default_system_prompt() -> SystemMessagePromptTemplate:
        return SystemMessagePromptTemplate.from_template(
            "PLEASE RESPOND ONLY WITH A SIGNLE FLOAT AND NO OTHER TEXT EXPLANATION\n You are a strict judge that is called on to rank a response based on given criteria.\
                    You must respond with your ranking by providing a single float within the range [0, 1], 0 being very bad response and 1 being very good response."
        )

    @staticmethod
    def get_default_prompt() -> ChatPromptTemplate:
        human_template = 'Given this based_on "{rl_chain_selected_based_on}" as the most important attribute, rank how good or bad this text is: "{llm_response}".'
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        default_system_prompt = AutoSelectionScorer.get_default_system_prompt()
        chat_prompt = ChatPromptTemplate.from_messages(
            [default_system_prompt, human_message_prompt]
        )
        return chat_prompt

    @root_validator(pre=True)
    def set_prompt_and_llm_chain(cls, values):
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

    def score_response(self, inputs: Dict[str, Any], llm_response: str) -> float:
        ranking = self.llm_chain.predict(llm_response=llm_response, **inputs)
        ranking = ranking.strip()
        try:
            resp = float(ranking)
            return resp
        except Exception as e:
            raise RuntimeError(
                f"The llm did not manage to rank the response as expected, there is always the option to try again or tweak the reward prompt. Error: {e}"
            )


class RLChain(Chain):
    """
    RLChain class that utilizes the Vowpal Wabbit (VW) model for personalization.

    Attributes:
        model_loading (bool, optional): If set to True, the chain will attempt to load an existing VW model from the latest checkpoint file in the {model_save_dir} directory (current directory if none specified). If set to False, it will start training from scratch, potentially overwriting existing files. Defaults to True.
        large_action_spaces (bool, optional): If set to True and vw_cmd has not been specified in the constructor, it will enable large action spaces
        vw_cmd (List[str], optional): Advanced users can set the VW command line to whatever they want, as long as it is compatible with the Type that is specified (Type Enum)
        model_save_dir (str, optional): The directory to save the VW model to. Defaults to the current directory.
        selection_scorer (SelectionScorer): If set, the chain will check the response using the provided selection_scorer and the VW model will be updated with the result. Defaults to None.

    Notes:
        The class creates a VW model instance using the provided arguments. Before the chain object is destroyed the save_progress() function can be called. If it is called, the learned VW model is saved to a file in the current directory named `model-<checkpoint>.vw`. Checkpoints start at 1 and increment monotonically.
        When making predictions, VW is first called to choose action(s) which are then passed into the prompt with the key `{actions}`. After action selection, the LLM (Language Model) is called with the prompt populated by the chosen action(s), and the response is returned.
    """

    llm_chain: Chain

    output_key: str = "result"  #: :meta private:
    prompt: BasePromptTemplate
    selection_scorer: Union[SelectionScorer, None]
    policy: Optional[Policy]
    auto_embed: bool = True
    selected_input_key = "rl_chain_selected"
    selected_based_on_input_key = "rl_chain_selected_based_on"
    metrics: Optional[MetricsTracker] = None

    def __init__(
        self,
        feature_embedder: Embedder,
        model_save_dir="./",
        reset_model=False,
        vw_cmd=None,
        policy=VwPolicy,
        vw_logs: Optional[Union[str, os.PathLike]] = None,
        metrics_step=-1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.selection_scorer is None:
            logger.warning(
                "No response validator provided, which means that no reinforcement learning will be done in the RL chain unless update_with_delayed_score is called."
            )
        self.policy = policy(
            model_repo=ModelRepository(
                model_save_dir, with_history=True, reset=reset_model
            ),
            vw_cmd=vw_cmd or [],
            feature_embedder=feature_embedder,
            vw_logger=VwLogger(vw_logs),
        )
        self.metrics = MetricsTracker(step=metrics_step)

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

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        super()._validate_inputs(inputs)
        if (
            self.selected_input_key in inputs.keys()
            or self.selected_based_on_input_key in inputs.keys()
        ):
            raise ValueError(
                f"The rl chain does not accept '{self.selected_input_key}' or '{self.selected_based_on_input_key}' as input keys, they are reserved for internal use during auto reward."
            )

    @abstractmethod
    def _call_before_predict(self, inputs: Dict[str, Any]) -> Event:
        pass

    @abstractmethod
    def _call_after_predict_before_llm(
        self, inputs: Dict[str, Any], event: Event, prediction: Any
    ) -> Tuple[Dict[str, Any], Event]:
        pass

    @abstractmethod
    def _call_after_llm_before_scoring(
        self, llm_response: str, event: Event
    ) -> Tuple[Dict[str, Any], Event]:
        pass

    @abstractmethod
    def _call_after_scoring_before_learning(
        self, event: Event, score: Optional[float]
    ) -> Event:
        pass

    def update_with_delayed_score(
        self, score: float, event: Event, force_score=False
    ) -> None:
        """
        Learn will be called with the score specified and the actions/embeddings/etc stored in event

        Will raise an error if selection_scorer is set, and force_score=True was not provided during the method call
        """
        if self.selection_scorer and not force_score:
            raise RuntimeError(
                "The selection scorer is set, and force_score was not set to True. Please set force_score=True to use this function."
            )
        self.metrics.on_feedback(score)
        self._call_after_scoring_before_learning(event=event, score=score)
        self.policy.learn(event=event)
        self.policy.log(event=event)

    def set_auto_embed(self, auto_embed: bool) -> None:
        """
        Set whether the chain should auto embed the inputs or not. If set to False, the inputs will not be embedded and the user will need to embed the inputs themselves before calling run.

        Args:
            auto_embed (bool): Whether the chain should auto embed the inputs or not.
        """
        self.auto_embed = auto_embed

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        if self.auto_embed:
            inputs = prepare_inputs_for_autoembed(inputs=inputs)

        event = self._call_before_predict(inputs=inputs)
        prediction = self.policy.predict(event=event)
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
            if self.selection_scorer:
                score = self.selection_scorer.score_response(
                    inputs=next_chain_inputs, llm_response=output
                )
        except Exception as e:
            logger.info(
                f"The LLM was not able to rank and the chain was not able to adjust to this response, error: {e}"
            )
        self.metrics.on_feedback(score)
        event = self._call_after_scoring_before_learning(score=score, event=event)
        self.policy.learn(event=event)
        self.policy.log(event=event)

        return {self.output_key: {"response": output, "selection_metadata": event}}

    def save_progress(self) -> None:
        """
        This function should be called whenever there is a need to save the progress of the VW (Vowpal Wabbit) model within the chain. It saves the current state of the VW model to a file.

        File Naming Convention:
          The file will be named using the pattern `model-<checkpoint>.vw`, where `<checkpoint>` is a monotonically increasing number. The numbering starts from 1, and increments by 1 for each subsequent save. If there are already saved checkpoints, the number used for `<checkpoint>` will be the next in the sequence.

        Example:
            If there are already two saved checkpoints, `model-1.vw` and `model-2.vw`, the next time this function is called, it will save the model as `model-3.vw`.

        Note:
            Be cautious when deleting or renaming checkpoint files manually, as this could cause the function to reuse checkpoint numbers.
        """
        self.policy.save()

    @property
    def _chain_type(self) -> str:
        return "llm_personalizer_chain"


def is_stringtype_instance(item: Any) -> bool:
    """Helper function to check if an item is a string."""
    return isinstance(item, str) or (
        isinstance(item, _Embed) and isinstance(item.value, str)
    )


def embed_string_type(
    item: Union[str, _Embed], model: Any, namespace: Optional[str] = None
) -> Dict[str, str]:
    """Helper function to embed a string or an _Embed object."""
    join_char = ""
    keep_str = ""
    if isinstance(item, _Embed):
        encoded = model.encode(item.value)
        join_char = " "
        if item.keep:
            keep_str = item.value.replace(" ", "_") + " "
    elif isinstance(item, str):
        encoded = item.replace(" ", "_")
        join_char = ""
    else:
        raise ValueError(f"Unsupported type {type(item)} for embedding")

    if namespace is None:
        raise ValueError(
            "The default namespace must be provided when embedding a string or _Embed object."
        )

    return {namespace: keep_str + join_char.join(map(str, encoded))}


def embed_dict_type(item: Dict, model: Any) -> Dict[str, Union[str, List[str]]]:
    """Helper function to embed a dictionary item."""
    inner_dict = {}
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
    ret_list = []
    for embed_item in item:
        if isinstance(embed_item, dict):
            ret_list.append(embed_dict_type(embed_item, model))
        else:
            ret_list.append(embed_string_type(embed_item, model, namespace))
    return ret_list


def embed(
    to_embed: Union[
        Union(str, _Embed(str)), Dict, List[Union(str, _Embed(str))], List[Dict]
    ],
    model: Any,
    namespace: Optional[str] = None,
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Embeds the actions or context using the SentenceTransformer model

    Attributes:
        to_embed: (Union[Union(str, _Embed(str)), Dict, List[Union(str, _Embed(str))], List[Dict]], required) The text to be embedded, either a string, a list of strings or a dictionary or a list of dictionaries.
        namespace: (str, optional) The default namespace to use when dictionary or list of dictionaries not provided.
        model: (Any, required) The model to use for embedding
    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary has the namespace as the key and the embedded string as the value
    """
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
