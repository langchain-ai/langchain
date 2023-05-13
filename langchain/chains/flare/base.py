""""""
from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import Field, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.flare.prompts import PROMPT, QUESTION_GENERATOR_PROMPT
from langchain.llms import OpenAI
from langchain.prompts import BasePromptTemplate
from langchain.schema import BaseRetriever, Generation


class ResponseChain(LLMChain):
    """"""

    prompt: BasePromptTemplate = PROMPT

    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        prompt = values["prompt"]
        prompt_inputs = set(prompt.input_variables)
        if prompt_inputs.difference(["user_input", "context", "response"]):
            raise ValueError
        if prompt.output_parser is None:
            raise ValueError
        return values

    @property
    def input_keys(self) -> List[str]:
        return ["user_input", "context", "response"]

    def generate_tokens_and_log_probs(
        self,
        _input: Dict[str, Any],
        *,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[Sequence[str], Sequence[float]]:
        llm_result = self.generate([_input], run_manager=run_manager)
        return self._extract_tokens_and_log_probs(llm_result.generations[0])

    @abstractmethod
    def _extract_tokens_and_log_probs(
        self, generations: List[Generation]
    ) -> Tuple[Sequence[str], Sequence[float]]:
        """"""


class _OpenAIResponseChain(ResponseChain):
    llm: OpenAI = Field(
        default_factory=lambda: OpenAI(max_tokens=64, model_kwargs={"logprobs": 1})
    )
    """"""

    def _extract_tokens_and_log_probs(
        self, generations: List[Generation]
    ) -> Tuple[Sequence[str], Sequence[float]]:
        tokens = []
        log_probs = []
        for gen in generations:
            if gen.generation_info is None:
                raise ValueError
            tokens.extend(gen.generation_info["logprobs"]["tokens"])
            log_probs.extend(gen.generation_info["logprobs"]["token_logprobs"])
        return tokens, log_probs


class QuestionGeneratorChain(LLMChain):
    prompt: BasePromptTemplate = QUESTION_GENERATOR_PROMPT

    @property
    def input_keys(self) -> List[str]:
        return ["user_input", "context", "response"]


def _low_confidence_spans(
    tokens: Sequence[str],
    log_probs: Sequence[float],
    min_prob: float,
    min_token_gap: int,
    num_pad_tokens: int,
) -> List[str]:
    low_idx = np.where(np.exp(log_probs) < min_prob)[0]
    low_idx = [i for i in low_idx if re.search(r"\w", tokens[i])]
    if len(low_idx) == 0:
        return []
    spans = [[low_idx[0], low_idx[0] + num_pad_tokens + 1]]
    for i, idx in enumerate(low_idx[1:]):
        end = idx + num_pad_tokens + 1
        if idx - low_idx[i] < min_token_gap:
            spans[-1][1] = end
        else:
            spans.append([idx, end])
    return ["".join(tokens[start:end]) for start, end in spans]


class FlareChain(Chain):
    """"""

    question_generator_chain: QuestionGeneratorChain
    """"""
    response_chain: ResponseChain = Field(default_factory=_OpenAIResponseChain)
    """"""
    retriever: BaseRetriever
    """"""
    min_prob: float = 0.1
    """"""
    min_token_gap: int = 5
    """"""
    num_pad_tokens: int = 2
    """"""
    max_iter: int = 10
    """"""

    @property
    def input_keys(self) -> List[str]:
        return ["user_input"]

    @property
    def output_keys(self) -> List[str]:
        return ["response"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        user_input = inputs[self.input_keys[0]]
        response = ""
        for _ in range(self.max_iter):
            _input = {"user_input": user_input, "context": "", "response": response}
            tokens, log_probs = self.response_chain.generate_tokens_and_log_probs(
                _input, run_manager=_run_manager
            )
            low_confidence_spans = _low_confidence_spans(
                tokens,
                log_probs,
                self.min_prob,
                self.min_token_gap,
                self.num_pad_tokens,
            )
            if not low_confidence_spans:
                response += "".join(tokens)
                continue
            question_gen_inputs = [
                {
                    "user_input": user_input,
                    "current_response": response,
                    "uncertain_span": span,
                }
                for span in low_confidence_spans
            ]
            question_gen_outputs = self.question_generator_chain.apply(
                question_gen_inputs, callbacks=callbacks
            )
            docs = []
            for output in question_gen_outputs:
                question = output[self.question_generator_chain.output_keys[0]]
                docs.extend(self.retriever.get_relevant_documents(question))
            context = "\n\n".join(d.page_content for d in docs)
            marginal, finished = self.response_chain.predict_and_parse(
                user_input=user_input,
                context=context,
                response=response,
                callbacks=callbacks,
            )
            response += marginal
            if finished:
                break
        return {self.output_keys[0]: response}

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> FlareChain:
        question_gen_chain = QuestionGeneratorChain(llm=llm)
        return cls(question_generator_chain=question_gen_chain, **kwargs)
