from __future__ import annotations

from . import rl_chain_base as base

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer
from langchain.prompts import BasePromptTemplate

import logging

logger = logging.getLogger(__name__)

# sentinel object used to distinguish between user didn't supply anything or user explicitly supplied None
SENTINEL = object()


class PickBestFeatureEmbedder(base.Embedder):
    """
    Contextual Bandit Text Embedder class that embeds the based_on and to_select_from into a format that can be used by VW
    
    Attributes:
        model name (Any, optional): The type of embeddings to be used for feature representation. Defaults to BERT SentenceTransformer.
    """

    def __init__(self, model: Optional[Any] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if model is None:
            model = SentenceTransformer("bert-base-nli-mean-tokens")

        self.model = model

    def format(self, event: PickBest.Event) -> str:
        """
        Converts the based_on and to_select_from into a format that can be used by VW
        """

        cost = None
        if event.selected:
            chosen_action = event.selected.index
            cost = (
                -1.0 * event.selected.score
                if event.selected.score is not None
                else None
            )
            prob = event.selected.probability

        context_emb = base.embed(event.based_on, self.model) if event.based_on else None
        to_select_from_var_name, to_select_from = next(
            iter(event.to_select_from.items()), (None, None)
        )
        action_embs = (
            base.embed(to_select_from, self.model, to_select_from_var_name)
            if event.to_select_from
            else None
        )

        if not context_emb or not action_embs:
            raise ValueError(
                "Context and to_select_from must be provided in the inputs dictionary"
            )

        example_string = ""
        example_string += f"shared "
        for context_item in context_emb:
            for ns, based_on in context_item.items():
                example_string += f"|{ns} {' '.join(based_on) if isinstance(based_on, list) else based_on} "
        example_string += "\n"

        for i, action in enumerate(action_embs):
            if cost is not None and chosen_action == i:
                example_string += f"{chosen_action}:{cost}:{prob} "
            for ns, action_embedding in action.items():
                example_string += f"|{ns} {' '.join(action_embedding) if isinstance(action_embedding, list) else action_embedding} "
            example_string += "\n"
        # Strip the last newline
        return example_string[:-1]


class PickBest(base.RLChain):
    """
    PickBest class that utilizes the Vowpal Wabbit (VW) model for personalization.

    The Chain is initialized with a set of potential to_select_from. For each call to the Chain, a specific action will be chosen based on an input based_on.
    This chosen action is then passed to the prompt that will be utilized in the subsequent call to the LLM (Language Model).

    The flow of this chain is:
    - Chain is initialized
    - Chain is called input containing the based_on and the List of potential to_select_from
    - Chain chooses an action based on the based_on
    - Chain calls the LLM with the chosen action
    - LLM returns a response
    - If the selection_scorer is specified, the response is checked against the selection_scorer
    - The internal model will be updated with the based_on, action, and reward of the response (how good or bad the response was)
    - The response is returned

    input dictionary expects:
        - at least one variable wrapped in BasedOn which will be the based_on to use for personalization
        - one variable of a list wrapped in ToSelectFrom which will be the list of to_select_from for the Vowpal Wabbit model to choose from.
            This list can either be a List of str's or a List of Dict's.
                - Actions provided as a list of strings e.g. to_select_from = ["action1", "action2", "action3"]
                - If to_select_from are provided as a list of dictionaries, each action should be a dictionary where the keys are namespace names and the values are the corresponding action strings e.g. to_select_from = [{"namespace1": "action1", "namespace2": "action2"}, {"namespace1": "action3", "namespace2": "action4"}]
    Extends:
        RLChain

    Attributes:
        feature_embedder: (PickBestFeatureEmbedder, optional) The text embedder to use for embedding the based_on and the to_select_from. If not provided, a default embedder is used.
    """

    class Selected(base.Selected):
        index: Optional[int]
        probability: Optional[float]
        score: Optional[float]

        def __init__(
            self,
            index: Optional[int] = None,
            probability: Optional[float] = None,
            score: Optional[float] = None,
        ):
            self.index = index
            self.probability = probability
            self.score = score

    class Event(base.Event):
        def __init__(
            self,
            inputs: Dict[str, Any],
            to_select_from: Dict[str, Any],
            based_on: Dict[str, Any],
            selected: Optional[PickBest.Selected] = None,
        ):
            super().__init__(inputs=inputs, selected=selected)
            self.to_select_from = to_select_from
            self.based_on = based_on

    def __init__(
        self,
        feature_embedder: Optional[PickBestFeatureEmbedder] = None,
        *args,
        **kwargs,
    ):
        vw_cmd = kwargs.get("vw_cmd", [])
        if not vw_cmd:
            vw_cmd = [
                "--cb_explore_adf",
                "--quiet",
                "--interactions=::",
                "--coin",
                "--epsilon=0.2",
            ]
        else:
            if "--cb_explore_adf" not in vw_cmd:
                raise ValueError(
                    "If vw_cmd is specified, it must include --cb_explore_adf"
                )

        kwargs["vw_cmd"] = vw_cmd
        if not feature_embedder:
            feature_embedder = PickBestFeatureEmbedder()

        super().__init__(feature_embedder=feature_embedder, *args, **kwargs)

    def _call_before_predict(self, inputs: Dict[str, Any]) -> PickBest.Event:
        context, actions = base.get_based_on_and_to_select_from(inputs=inputs)
        if not actions:
            raise ValueError(
                "No variables using 'ToSelectFrom' found in the inputs. Please include at least one variable containing a list to select from."
            )

        if len(list(actions.values())) > 1:
            raise ValueError(
                "Only one variable using 'ToSelectFrom' can be provided in the inputs for the PickBest chain. Please provide only one variable containing a list to select from."
            )

        if not context:
            raise ValueError(
                "No variables using 'BasedOn' found in the inputs. Please include at least one variable containing information to base the selected of ToSelectFrom on."
            )

        event = PickBest.Event(inputs=inputs, to_select_from=actions, based_on=context)
        return event

    def _call_after_predict_before_llm(
        self, inputs: Dict[str, Any], event: Event, prediction: List[Tuple[int, float]]
    ) -> Tuple[Dict[str, Any], PickBest.Event]:
        prob_sum = sum(prob for _, prob in prediction)
        probabilities = [prob / prob_sum for _, prob in prediction]
        ## sample from the pmf
        sampled_index = np.random.choice(len(prediction), p=probabilities)
        sampled_ap = prediction[sampled_index]
        sampled_action = sampled_ap[0]
        sampled_prob = sampled_ap[1]
        selected = PickBest.Selected(index=sampled_action, probability=sampled_prob)
        event.selected = selected

        # only one key, value pair in event.to_select_from
        key, value = next(iter(event.to_select_from.items()))
        next_chain_inputs = inputs.copy()
        next_chain_inputs.update({key: value[event.selected.index]})
        return next_chain_inputs, event

    def _call_after_llm_before_scoring(
        self, llm_response: str, event: PickBest.Event
    ) -> Tuple[Dict[str, Any], PickBest.Event]:
        next_chain_inputs = event.inputs.copy()
        # only one key, value pair in event.to_select_from
        value = next(iter(event.to_select_from.values()))
        next_chain_inputs.update(
            {
                self.selected_based_on_input_key: str(event.based_on),
                self.selected_input_key: value[event.selected.index],
            }
        )
        return next_chain_inputs, event

    def _call_after_scoring_before_learning(
        self, event: PickBest.Event, score: Optional[float]
    ) -> Event:
        event.selected.score = score
        return event

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        When chain.run() is called with the given inputs, this function is called. It is responsible for calling the VW model to choose an action (ToSelectFrom) based on the (BasedOn) based_on, and then calling the LLM (Language Model) with the chosen action to generate a response.

        Attributes:
            inputs: (Dict, required) The inputs to the chain. The inputs must contain a input variables that are wrapped in BasedOn and ToSelectFrom. BasedOn is the based_on that will be used for selecting an ToSelectFrom action that will be passed to the LLM prompt.
            run_manager: (CallbackManagerForChainRun, optional) The callback manager to use for this run. If not provided, a default callback manager is used.
            
        Returns:
            A dictionary containing:
                - `response`: The response generated by the LLM (Language Model).
                - `selection_metadata`: A Event object containing all the information needed to learn the reward for the chosen action at a later point. If an automatic selection_scorer is not provided, then this object can be used at a later point with the `update_with_delayed_score()` function to learn the delayed reward and update the Vowpal Wabbit model.
                    - the `score` in the `selection_metadata` object is set to None if an automatic selection_scorer is not provided or if the selection_scorer failed (e.g. LLM timeout or LLM failed to rank correctly).
        """
        return super()._call(run_manager=run_manager, inputs=inputs)

    @property
    def _chain_type(self) -> str:
        return "rl_chain_pick_best"

    @classmethod
    def from_chain(
        cls,
        llm_chain: Chain,
        prompt: BasePromptTemplate,
        selection_scorer=SENTINEL,
        **kwargs: Any,
    ):
        if selection_scorer is SENTINEL:
            selection_scorer = base.AutoSelectionScorer(llm=llm_chain.llm)
        return PickBest(
            llm_chain=llm_chain,
            prompt=prompt,
            selection_scorer=selection_scorer,
            **kwargs,
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate,
        selection_scorer=SENTINEL,
        **kwargs: Any,
    ):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return PickBest.from_chain(
            llm_chain=llm_chain,
            prompt=prompt,
            selection_scorer=selection_scorer,
            **kwargs,
        )
