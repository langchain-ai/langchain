from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, Optional

from langchain_core.callbacks import (
    CallbackManagerForChainRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from pydantic import Field

from langchain.chains.base import Chain
from langchain.chains.flare.prompts import (
    PROMPT,
    QUESTION_GENERATOR_PROMPT,
    FinishedOutputParser,
)
from langchain.chains.llm import LLMChain

logger = logging.getLogger(__name__)


def _extract_tokens_and_log_probs(response: AIMessage) -> tuple[list[str], list[float]]:
    """Extract tokens and log probabilities from chat model response."""
    tokens = []
    log_probs = []
    for token in response.response_metadata["logprobs"]["content"]:
        tokens.append(token["token"])
        log_probs.append(token["logprob"])
    return tokens, log_probs


class QuestionGeneratorChain(LLMChain):
    """Chain that generates questions from uncertain spans."""

    prompt: BasePromptTemplate = QUESTION_GENERATOR_PROMPT
    """Prompt template for the chain."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def input_keys(self) -> list[str]:
        """Input keys for the chain."""
        return ["user_input", "context", "response"]


def _low_confidence_spans(
    tokens: Sequence[str],
    log_probs: Sequence[float],
    min_prob: float,
    min_token_gap: int,
    num_pad_tokens: int,
) -> list[str]:
    try:
        import numpy as np

        _low_idx = np.where(np.exp(log_probs) < min_prob)[0]
    except ImportError:
        logger.warning(
            "NumPy not found in the current Python environment. FlareChain will use a "
            "pure Python implementation for internal calculations, which may "
            "significantly impact performance, especially for large datasets. For "
            "optimal speed and efficiency, consider installing NumPy: pip install "
            "numpy",
        )
        import math

        _low_idx = [  # type: ignore[assignment]
            idx
            for idx, log_prob in enumerate(log_probs)
            if math.exp(log_prob) < min_prob
        ]
    low_idx = [i for i in _low_idx if re.search(r"\w", tokens[i])]
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
    """Chain that combines a retriever, a question generator,
    and a response generator.

    See [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983) paper.
    """

    question_generator_chain: Runnable
    """Chain that generates questions from uncertain spans."""
    response_chain: Runnable
    """Chain that generates responses from user input and context."""
    output_parser: FinishedOutputParser = Field(default_factory=FinishedOutputParser)
    """Parser that determines whether the chain is finished."""
    retriever: BaseRetriever
    """Retriever that retrieves relevant documents from a user input."""
    min_prob: float = 0.2
    """Minimum probability for a token to be considered low confidence."""
    min_token_gap: int = 5
    """Minimum number of tokens between two low confidence spans."""
    num_pad_tokens: int = 2
    """Number of tokens to pad around a low confidence span."""
    max_iter: int = 10
    """Maximum number of iterations."""
    start_with_retrieval: bool = True
    """Whether to start with retrieval."""

    @property
    def input_keys(self) -> list[str]:
        """Input keys for the chain."""
        return ["user_input"]

    @property
    def output_keys(self) -> list[str]:
        """Output keys for the chain."""
        return ["response"]

    def _do_generation(
        self,
        questions: list[str],
        user_input: str,
        response: str,
        _run_manager: CallbackManagerForChainRun,
    ) -> tuple[str, bool]:
        callbacks = _run_manager.get_child()
        docs = []
        for question in questions:
            docs.extend(self.retriever.invoke(question))
        context = "\n\n".join(d.page_content for d in docs)
        result = self.response_chain.invoke(
            {
                "user_input": user_input,
                "context": context,
                "response": response,
            },
            {"callbacks": callbacks},
        )
        if isinstance(result, AIMessage):
            result = result.content
        marginal, finished = self.output_parser.parse(result)
        return marginal, finished

    def _do_retrieval(
        self,
        low_confidence_spans: list[str],
        _run_manager: CallbackManagerForChainRun,
        user_input: str,
        response: str,
        initial_response: str,
    ) -> tuple[str, bool]:
        question_gen_inputs = [
            {
                "user_input": user_input,
                "current_response": initial_response,
                "uncertain_span": span,
            }
            for span in low_confidence_spans
        ]
        callbacks = _run_manager.get_child()
        if isinstance(self.question_generator_chain, LLMChain):
            question_gen_outputs = self.question_generator_chain.apply(
                question_gen_inputs,
                callbacks=callbacks,
            )
            questions = [
                output[self.question_generator_chain.output_keys[0]]
                for output in question_gen_outputs
            ]
        else:
            questions = self.question_generator_chain.batch(
                question_gen_inputs,
                config={"callbacks": callbacks},
            )
        _run_manager.on_text(
            f"Generated Questions: {questions}",
            color="yellow",
            end="\n",
        )
        return self._do_generation(questions, user_input, response, _run_manager)

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        user_input = inputs[self.input_keys[0]]

        response = ""

        for _i in range(self.max_iter):
            _run_manager.on_text(
                f"Current Response: {response}",
                color="blue",
                end="\n",
            )
            _input = {"user_input": user_input, "context": "", "response": response}
            tokens, log_probs = _extract_tokens_and_log_probs(
                self.response_chain.invoke(
                    _input,
                    {"callbacks": _run_manager.get_child()},
                ),
            )
            low_confidence_spans = _low_confidence_spans(
                tokens,
                log_probs,
                self.min_prob,
                self.min_token_gap,
                self.num_pad_tokens,
            )
            initial_response = response.strip() + " " + "".join(tokens)
            if not low_confidence_spans:
                response = initial_response
                final_response, finished = self.output_parser.parse(response)
                if finished:
                    return {self.output_keys[0]: final_response}
                continue

            marginal, finished = self._do_retrieval(
                low_confidence_spans,
                _run_manager,
                user_input,
                response,
                initial_response,
            )
            response = response.strip() + " " + marginal
            if finished:
                break
        return {self.output_keys[0]: response}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        max_generation_len: int = 32,
        **kwargs: Any,
    ) -> FlareChain:
        """Creates a FlareChain from a language model.

        Args:
            llm: Language model to use.
            max_generation_len: Maximum length of the generated response.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            FlareChain class with the given language model.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            msg = (
                "OpenAI is required for FlareChain. "
                "Please install langchain-openai."
                "pip install langchain-openai"
            )
            raise ImportError(msg) from e
        llm = ChatOpenAI(
            max_completion_tokens=max_generation_len,
            logprobs=True,
            temperature=0,
        )
        response_chain = PROMPT | llm
        question_gen_chain = QUESTION_GENERATOR_PROMPT | llm | StrOutputParser()
        return cls(
            question_generator_chain=question_gen_chain,
            response_chain=response_chain,
            **kwargs,
        )
