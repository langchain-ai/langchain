from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import langchain.chains.rl_chain.base as base
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.prompts import BasePromptTemplate

logger = logging.getLogger(__name__)

# sentinel object used to distinguish between
# user didn't supply anything or user explicitly supplied None
SENTINEL = object()


class PickBestSelected(base.Selected):
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


class PickBestEvent(base.Event[PickBestSelected]):
    def __init__(
        self,
        inputs: Dict[str, Any],
        to_select_from: Dict[str, Any],
        based_on: Dict[str, Any],
        selected: Optional[PickBestSelected] = None,
    ):
        super().__init__(inputs=inputs, selected=selected)
        self.to_select_from = to_select_from
        self.based_on = based_on


class PickBestFeatureEmbedder(base.Embedder[PickBestEvent]):
    """
    Text Embedder class that embeds the `BasedOn` and `ToSelectFrom` inputs into a format that can be used by the learning policy

    Attributes:
        model name (Any, optional): The type of embeddings to be used for feature representation. Defaults to BERT SentenceTransformer.
    """  # noqa E501

    def __init__(self, model: Optional[Any] = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("bert-base-nli-mean-tokens")

        self.model = model

    def format(self, event: PickBestEvent) -> str:
        """
        Converts the `BasedOn` and `ToSelectFrom` into a format that can be used by VW
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
            (
                base.embed(to_select_from, self.model, to_select_from_var_name)
                if event.to_select_from
                else None
            )
            if to_select_from
            else None
        )

        if not context_emb or not action_embs:
            raise ValueError(
                "Context and to_select_from must be provided in the inputs dictionary"
            )

        example_string = ""
        example_string += "shared "
        for context_item in context_emb:
            for ns, based_on in context_item.items():
                e = " ".join(based_on) if isinstance(based_on, list) else based_on
                example_string += f"|{ns} {e} "
        example_string += "\n"

        for i, action in enumerate(action_embs):
            if cost is not None and chosen_action == i:
                example_string += f"{chosen_action}:{cost}:{prob} "
            for ns, action_embedding in action.items():
                e = (
                    " ".join(action_embedding)
                    if isinstance(action_embedding, list)
                    else action_embedding
                )
                example_string += f"|{ns} {e} "
            example_string += "\n"
        # Strip the last newline
        return example_string[:-1]


class PickBest(base.RLChain[PickBestEvent]):
    """
    `PickBest` is a class designed to leverage the Vowpal Wabbit (VW) model for reinforcement learning with a context, with the goal of modifying the prompt before the LLM call.

    Each invocation of the chain's `run()` method should be equipped with a set of potential actions (`ToSelectFrom`) and will result in the selection of a specific action based on the `BasedOn` input. This chosen action then informs the LLM (Language Model) prompt for the subsequent response generation.

    The standard operation flow of this Chain includes:
        1. The Chain is invoked with inputs containing the `BasedOn` criteria and a list of potential actions (`ToSelectFrom`).
        2. An action is selected based on the `BasedOn` input.
        3. The LLM is called with the dynamic prompt, producing a response.
        4. If a `selection_scorer` is provided, it is used to score the selection.
        5. The internal Vowpal Wabbit model is updated with the `BasedOn` input, the chosen `ToSelectFrom` action, and the resulting score from the scorer.
        6. The final response is returned.

    Expected input dictionary format:
        - At least one variable encapsulated within `BasedOn` to serve as the selection criteria.
        - A single list variable within `ToSelectFrom`, representing potential actions for the VW model. This list can take the form of:
            - A list of strings, e.g., `action = ToSelectFrom(["action1", "action2", "action3"])`
            - A list of list of strings e.g. `action = ToSelectFrom([["action1", "another identifier of action1"], ["action2", "another identifier of action2"]])`
            - A list of dictionaries, where each dictionary represents an action with namespace names as keys and corresponding action strings as values. For instance, `action = ToSelectFrom([{"namespace1": ["action1", "another identifier of action1"], "namespace2": "action2"}, {"namespace1": "action3", "namespace2": "action4"}])`.

    Extends:
        RLChain

    Attributes:
        feature_embedder (PickBestFeatureEmbedder, optional): Is an advanced attribute. Responsible for embedding the `BasedOn` and `ToSelectFrom` inputs. If omitted, a default embedder is utilized.
    """  # noqa E501

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        vw_cmd = kwargs.get("vw_cmd", [])
        if not vw_cmd:
            vw_cmd = [
                "--cb_explore_adf",
                "--quiet",
                "--interactions=::",
                "--coin",
                "--squarecb",
            ]
        else:
            if "--cb_explore_adf" not in vw_cmd:
                raise ValueError(
                    "If vw_cmd is specified, it must include --cb_explore_adf"
                )
        kwargs["vw_cmd"] = vw_cmd

        feature_embedder = kwargs.get("feature_embedder", None)
        if not feature_embedder:
            feature_embedder = PickBestFeatureEmbedder()
        kwargs["feature_embedder"] = feature_embedder

        super().__init__(*args, **kwargs)

    def _call_before_predict(self, inputs: Dict[str, Any]) -> PickBestEvent:
        context, actions = base.get_based_on_and_to_select_from(inputs=inputs)
        if not actions:
            raise ValueError(
                "No variables using 'ToSelectFrom' found in the inputs. \
                    Please include at least one variable containing \
                        a list to select from."
            )

        if len(list(actions.values())) > 1:
            raise ValueError(
                "Only one variable using 'ToSelectFrom' can be provided in the inputs \
                    for the PickBest chain. Please provide only one variable \
                        containing a list to select from."
            )

        if not context:
            raise ValueError(
                "No variables using 'BasedOn' found in the inputs. \
                    Please include at least one variable containing information \
                        to base the selected of ToSelectFrom on."
            )

        event = PickBestEvent(inputs=inputs, to_select_from=actions, based_on=context)
        return event

    def _call_after_predict_before_llm(
        self,
        inputs: Dict[str, Any],
        event: PickBestEvent,
        prediction: List[Tuple[int, float]],
    ) -> Tuple[Dict[str, Any], PickBestEvent]:
        import numpy as np

        prob_sum = sum(prob for _, prob in prediction)
        probabilities = [prob / prob_sum for _, prob in prediction]
        ## sample from the pmf
        sampled_index = np.random.choice(len(prediction), p=probabilities)
        sampled_ap = prediction[sampled_index]
        sampled_action = sampled_ap[0]
        sampled_prob = sampled_ap[1]
        selected = PickBestSelected(index=sampled_action, probability=sampled_prob)
        event.selected = selected

        # only one key, value pair in event.to_select_from
        key, value = next(iter(event.to_select_from.items()))
        next_chain_inputs = inputs.copy()
        next_chain_inputs.update({key: value[event.selected.index]})
        return next_chain_inputs, event

    def _call_after_llm_before_scoring(
        self, llm_response: str, event: PickBestEvent
    ) -> Tuple[Dict[str, Any], PickBestEvent]:
        next_chain_inputs = event.inputs.copy()
        # only one key, value pair in event.to_select_from
        value = next(iter(event.to_select_from.values()))
        v = (
            value[event.selected.index]
            if event.selected
            else event.to_select_from.values()
        )
        next_chain_inputs.update(
            {
                self.selected_based_on_input_key: str(event.based_on),
                self.selected_input_key: v,
            }
        )
        return next_chain_inputs, event

    def _call_after_scoring_before_learning(
        self, event: PickBestEvent, score: Optional[float]
    ) -> PickBestEvent:
        if event.selected:
            event.selected.score = score
        return event

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        return super()._call(run_manager=run_manager, inputs=inputs)

    @property
    def _chain_type(self) -> str:
        return "rl_chain_pick_best"

    @classmethod
    def from_llm(
        cls: Type[PickBest],
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate,
        selection_scorer: Union[base.AutoSelectionScorer, object] = SENTINEL,
        **kwargs: Any,
    ) -> PickBest:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        if selection_scorer is SENTINEL:
            selection_scorer = base.AutoSelectionScorer(llm=llm_chain.llm)

        return PickBest(
            llm_chain=llm_chain,
            prompt=prompt,
            selection_scorer=selection_scorer,
            **kwargs,
        )
