"""Base classes for comparing the output of two models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra, Field

from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.evaluation.comparison.prompt import PROMPT, PROMPT_WITH_REFERENCE
from langchain.evaluation.schema import LLMEvalChain, PairwiseStringEvaluator
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import RUN_KEY, BaseOutputParser
from langchain.schema.language_model import BaseLanguageModel


class PairwiseStringResultOutputParser(BaseOutputParser[dict]):
    """A parser for the output of the PairwiseStringEvalChain.

    Attributes:
        _type (str): The type of the output parser.

    """

    @property
    def _type(self) -> str:
        """Return the type of the output parser.

        Returns:
            str: The type of the output parser.

        """
        return "pairwise_string_result"

    def parse(self, text: str) -> Any:
        """Parse the output text.

        Args:
            text (str): The output text to parse.

        Returns:
            Any: The parsed output.

        Raises:
            ValueError: If the verdict is invalid.

        """
        reasoning, verdict = text.strip().rsplit("\n", maxsplit=1)
        verdict = verdict.strip("[").strip("]")
        if verdict not in {"A", "B", "C"}:
            raise ValueError(
                f"Invalid verdict: {verdict}. "
                "Verdict must be one of 'A', 'B', or 'C'."
            )
        # C means the models are tied. Return 'None' meaning no preference
        verdict_ = None if verdict == "C" else verdict
        score = {
            "A": 1,
            "B": 0,
            None: 0.5,
        }.get(verdict_)
        return {
            "reasoning": reasoning,
            "value": verdict_,
            "score": score,
        }


class PairwiseStringEvalChain(PairwiseStringEvaluator, LLMEvalChain, LLMChain):
    """A chain for comparing two outputs, such as the outputs
     of two models, prompts, or outputs of a single model on similar inputs.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    Example:
        >>> from langchain.chat_models import ChatOpenAI
        >>> from langchain.evaluation.comparison import PairwiseStringEvalChain
        >>> llm = ChatOpenAI(temperature=0)
        >>> chain = PairwiseStringEvalChain.from_llm(llm=llm)
        >>> result = chain.evaluate_string_pairs(
        ...     input = "What is the chemical formula for water?",
        ...     prediction = "H2O",
        ...     prediction_b = (
        ...        "The chemical formula for water is H2O, which means"
        ...        " there are two hydrogen atoms and one oxygen atom."
        ...     reference = "The chemical formula for water is H2O.",
        ... )
        >>> print(result["text"])
        # {
        #    "value": "B",
        #    "comment": "Both responses accurately state"
        #       " that the chemical formula for water is H2O."
        #       " However, Response B provides additional information"
        # .     " by explaining what the formula means.\\n[[B]]"
        # }

    """

    output_key: str = "results"  #: :meta private:
    output_parser: BaseOutputParser = Field(
        default_factory=PairwiseStringResultOutputParser
    )

    class Config:
        """Configuration for the PairwiseStringEvalChain."""

        extra = Extra.ignore

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if the chain requires a reference, False otherwise.

        """
        return False

    @property
    def requires_input(self) -> bool:
        """Return whether the chain requires an input.

        Returns:
            bool: True if the chain requires an input, False otherwise.

        """
        return True

    @property
    def _skip_reference_warning(self) -> str:
        """Return the warning to show when reference is ignored.

        Returns:
            str: The warning to show when reference is ignored.

        """
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
            "\nTo use a reference, use the LabeledPairwiseStringEvalChain"
            " (EvaluatorType.LABELED_PAIRWISE_STRING) instead."
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> PairwiseStringEvalChain:
        """Initialize the PairwiseStringEvalChain from an LLM.

        Args:
            llm (BaseLanguageModel): The LLM to use.
            prompt (PromptTemplate, optional): The prompt to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            PairwiseStringEvalChain: The initialized PairwiseStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """
        expected_input_vars = {"prediction", "prediction_b", "input"}
        prompt_ = prompt or PROMPT
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
        return cls(llm=llm, prompt=prompt_, **kwargs)

    def _prepare_input(
        self,
        prediction: str,
        prediction_b: str,
        input: Optional[str],
        reference: Optional[str],
    ) -> dict:
        """Prepare the input for the chain.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            reference (str, optional): The reference string, if any.

        Returns:
            dict: The prepared input for the chain.

        """
        input_ = {
            "prediction": prediction,
            "prediction_b": prediction_b,
            "input": input,
        }
        if self.requires_reference:
            input_["reference"] = reference
        return input_

    def _prepare_output(self, result: dict) -> dict:
        """Prepare the output."""
        parsed = result[self.output_key]
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        return parsed

    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate whether output A is preferred to output B.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.

        """
        input_ = self._prepare_input(prediction, prediction_b, input, reference)
        result = self(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate whether output A is preferred to output B.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.

        """
        input_ = self._prepare_input(prediction, prediction_b, input, reference)
        result = await self.acall(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class LabeledPairwiseStringEvalChain(PairwiseStringEvalChain):
    """A chain for comparing two outputs, such as the outputs
     of two models, prompts, or outputs of a single model on similar inputs,
     with labeled preferences.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    """

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if the chain requires a reference, False otherwise.

        """
        return True

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> PairwiseStringEvalChain:
        """Initialize the LabeledPairwiseStringEvalChain from an LLM.

        Args:
            llm (BaseLanguageModel): The LLM to use.
            prompt (PromptTemplate, optional): The prompt to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            LabeledPairwiseStringEvalChain: The initialized LabeledPairwiseStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """  # noqa: E501
        expected_input_vars = {"prediction", "prediction_b", "input", "reference"}
        prompt_ = prompt or PROMPT_WITH_REFERENCE
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
        return cls(llm=llm, prompt=prompt_, **kwargs)
