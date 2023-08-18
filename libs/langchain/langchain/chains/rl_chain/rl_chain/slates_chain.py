from __future__ import annotations

from . import rl_chain_base as base
from langchain.prompts.prompt import PromptTemplate

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain

from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import chain
import random


from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer

# sentinel object used to distinguish between user didn't supply anything or user explicitly supplied None
SENTINEL = object()


class SlatesFeatureEmbedder(base.Embedder):
    """
    Slates Text Embedder class that embeds the context and actions and slates into a format that can be used by VW
    
    Attributes:
        model (Any, optional): The type of embeddings to be used for feature representation. Defaults to BERT Sentence Transformer
    """

    def __init__(self, model: Optional[Any] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if model is None:
            model = SentenceTransformer("bert-base-nli-mean-tokens")

        self.model = model

    def to_action_features(self, actions: Dict[str, Any]):
        def _str(embedding):
            return " ".join([f"{i}:{e}" for i, e in enumerate(embedding)])

        action_features = []
        for slot in actions.values():
            slot_features = []
            for action in slot:
                if isinstance(action, base._Embed) and action.keep:
                    feature = (
                        action.value.replace(" ", "_")
                        + " "
                        + _str(self.model.encode(action.value))
                    )
                elif isinstance(action, base._Embed):
                    feature = _str(self.model.encode(action.value))
                else:
                    feature = action.replace(" ", "_")
                slot_features.append(feature)
            action_features.append(slot_features)

        return action_features

    def format(self, event: SlatesPersonalizerChain.Event) -> str:
        action_features = self.to_action_features(event.to_select_from)

        cost = (
            -1.0 * event.selected.score
            if event.selected and event.selected.score is not None
            else ""
        )
        context_str = f"slates shared {cost} "

        if event.based_on:
            embedded_context = base.embed(event.based_on, self.model)
            for context_item in embedded_context:
                for ns, ctx in context_item.items():
                    context_str += (
                        f"|{ns} {' '.join(ctx) if isinstance(ctx, list) else ctx} "
                    )
        else:
            context_str += "|"  # empty context

        actions = chain.from_iterable(
            [
                [f"slates action {i} |Action {action}"]
                for i, slot in enumerate(action_features)
                for action in slot
            ]
        )
        ps = (
            [f"{a}:{p}" for a, p in event.selected.get_indexes_and_probabilities()]
            if event.selected
            else [""] * len(action_features)
        )
        slots = [f"slates slot {p} |" for p in ps]
        return "\n".join(list(chain.from_iterable([[context_str], actions, slots])))


class SlatesRandomPolicy(base.Policy):
    def __init__(self, feature_embedder: base.Embedder, *_, **__):
        self.feature_embedder = feature_embedder

    def predict(self, event: SlatesPersonalizerChain.Event) -> Any:
        return [
            [(random.randint(0, len(slot) - 1), 1.0 / len(slot))]
            for _, slot in event.to_select_from.items()
        ]

    def learn(self, event: SlatesPersonalizerChain.Event) -> Any:
        pass

    def log(self, event: SlatesPersonalizerChain.Event) -> Any:
        pass


class SlatesFirstChoicePolicy(base.Policy):
    def __init__(self, feature_embedder: base.Embedder, *_, **__):
        self.feature_embedder = feature_embedder

    def predict(self, event: SlatesPersonalizerChain.Event) -> Any:
        return [[(0, 1)] for _ in event.to_select_from]

    def learn(self, event: SlatesPersonalizerChain.Event) -> Any:
        pass

    def log(self, event: SlatesPersonalizerChain.Event) -> Any:
        pass


class SlatesPersonalizerChain(base.RLChain):
    class Selected(base.Selected):
        indexes: Optional[List[int]]
        probabilities: Optional[List[float]]
        score: Optional[float]

        def __init__(
            self,
            indexes: Optional[List[int]] = None,
            probabilities: Optional[List[float]] = None,
            score: Optional[float] = None,
        ):
            self.indexes = indexes
            self.probabilities = probabilities
            self.score = score

        def get_indexes_and_probabilities(self):
            return zip(self.indexes, self.probabilities)

    class Event(base.Event):
        def __init__(
            self,
            inputs: Dict[str, Any],
            to_select_from: Dict[str, Any],
            based_on: Dict[str, Any],
            selected: Optional[SlatesPersonalizerChain.Selected] = None,
        ):
            super().__init__(inputs=inputs, selected=selected)
            self.to_select_from = to_select_from
            self.based_on = based_on

    def __init__(
        self, feature_embedder: Optional[base.Embedder] = None, *args, **kwargs
    ):
        vw_cmd = kwargs.get("vw_cmd", [])
        if not vw_cmd:
            vw_cmd = [
                "--slates",
                "--quiet",
                "--interactions=::",
                "--coin",
                "--squarecb",
            ]
        else:
            if "--slates" not in vw_cmd:
                raise ValueError("If vw_cmd is specified, it must include --slates")

        kwargs["vw_cmd"] = vw_cmd

        if feature_embedder is None:
            feature_embedder = SlatesFeatureEmbedder()

        super().__init__(feature_embedder=feature_embedder, *args, **kwargs)

    def _call_before_predict(
        self, inputs: Dict[str, Any]
    ) -> SlatesPersonalizerChain.Event:
        context, actions = base.get_based_on_and_to_select_from(inputs=inputs)
        event = SlatesPersonalizerChain.Event(
            inputs=inputs, to_select_from=actions, based_on=context
        )
        return event

    def _call_after_predict_before_llm(
        self,
        inputs: Dict[str, Any],
        event: SlatesPersonalizerChain.Event,
        prediction: List[List[Tuple[int, float]]],
    ) -> Tuple[Dict[str, Any], SlatesPersonalizerChain.Event]:
        indexes = [p[0][0] for p in prediction]
        probabilities = [p[0][1] for p in prediction]
        selected = SlatesPersonalizerChain.Selected(
            indexes=indexes, probabilities=probabilities
        )
        event.selected = selected

        preds = {}
        for i, (j, a) in enumerate(
            zip(event.selected.indexes, event.to_select_from.values())
        ):
            preds[list(event.to_select_from.keys())[i]] = str(a[j])

        next_chain_inputs = inputs.copy()
        next_chain_inputs.update(preds)

        return next_chain_inputs, event

    def _call_after_llm_before_scoring(
        self, llm_response: str, event: SlatesPersonalizerChain.Event
    ) -> Tuple[Dict[str, Any], SlatesPersonalizerChain.Event]:
        next_chain_inputs = event.inputs.copy()
        next_chain_inputs.update(
            {
                self.selected_based_on_input_key: str(event.based_on),
                self.selected_input_key: str(event.to_select_from),
            }
        )
        return next_chain_inputs, event

    def _call_after_scoring_before_learning(
        self, event: Event, score: Optional[float]
    ) -> SlatesPersonalizerChain.Event:
        event.selected.score = score
        return event

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return super()._call(run_manager=run_manager, inputs=inputs)

    @property
    def _chain_type(self) -> str:
        return "llm_personalizer_chain"

    @classmethod
    def from_chain(
        cls,
        llm_chain: Chain,
        prompt: PromptTemplate,
        selection_scorer=SENTINEL,
        **kwargs: Any,
    ):
        if selection_scorer is SENTINEL:
            selection_scorer = base.AutoSelectionScorer(llm=llm_chain.llm)
        return SlatesPersonalizerChain(
            llm_chain=llm_chain,
            prompt=prompt,
            selection_scorer=selection_scorer,
            **kwargs,
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: PromptTemplate,
        selection_scorer=SENTINEL,
        **kwargs: Any,
    ):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return SlatesPersonalizerChain.from_chain(
            llm_chain=llm_chain,
            prompt=prompt,
            selection_scorer=selection_scorer,
            **kwargs,
        )
