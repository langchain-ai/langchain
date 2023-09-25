import logging
import re
from typing import Dict

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.evaluation import PairwiseStringEvalChain
from langchain.evaluation.comparison.llm_as_a_judge.prompt import COMPARISON_TEMPLATE
from langchain.schema import BaseOutputParser

logger = logging.getLogger(__name__)

_FIND_DOUBLE_BRACKETS = re.compile(r"\[\[(.*?)\]\]")


class LLMAsAJudgePairwiseOutputParser(BaseOutputParser[dict]):
    """A parser for the output of the LLM-as-a-judge eval chain.

    Attributes:
        _type (str): The type of the output parser.

    """

    _verdict_map: Dict[str, str] = {
        "A": "Win",
        "B": "Loss",
        "C": "Tie",
    }

    @property
    def _type(self) -> str:
        """Return the type of the output parser.

        Returns:
            str: The type of the output parser.

        """
        return "llm_as_a_judge_pairwise_string_result"

    def parse(self, text: str) -> Dict[str, str]:
        """Parse the output text.

        Args:
            text (str): The output text to parse.

        Returns:
            Dict: The parsed output.

        Raises:
            ValueError: If the verdict is invalid.

        """

        match = _FIND_DOUBLE_BRACKETS.search(text)

        if match:
            verdict = match.group(1)

        if not match or verdict not in {"A", "B", "C"}:
            raise ValueError(
                f"Invalid output: {text}. "
                "Output must contain a double bracketed string\
                 with the verdict 'A', 'B', or 'C'."
            )

        return {
            "reasoning": text,
            "verdict": self._verdict_map[verdict],
        }


class LLMAsAJudgePairwiseEvalChain(PairwiseStringEvalChain):
    """A chain for comparing two outputs, such as the outputs
    of two models, prompts, or outputs of a single model on similar inputs,
    with the "LLM-as-a-judge" comparison method. This method achieves 85% \
    agreement with humans when using GPT-4.

    The verdict can be one of "Win", "Loss", or "Tie". With win meaning
    prediction A is better than prediction B.


    Example:
        >>> from langchain.chat_models import ChatOpenAI
        >>> from langchain.evaluation.comparison.llm_as_a_judge import LLMAsAJudgePairwiseEvalChain
        >>> llm = ChatOpenAI(temperature=0, model="gpt-4")
        >>> chain = LLMAsAJudgePairwiseEvalChain.from_llm(llm=llm)
        >>> result = chain.evaluate_string_pairs(
        ...     input = "What is the chemical formula for water?",
        ...     prediction = "H2O",
        ...     prediction_b = (
        ...        "The chemical formula for water is H2O, which means"
        ...        " there are two hydrogen atoms and one oxygen atom."
        ... )
        >>> print(result)
        # {
        #    "verdict": "Win",
        #    "reasoning": ""Both Assistant A and Assistant B "
        #       "accurately state that the chemical formula for water is H2O."
        #       "However, Assistant A provided a more detailed response by explaining what the formula H2O means,"
        #       " i.e., it consists of two hydrogen atoms and one oxygen atom."
        #       " This additional information could be helpful to the user, especially"
        #       " if they are not familiar with chemical formulas."
        #       " Therefore, Assistant A's response is more comprehensive and informative."
        #       "Final Verdict: [[A]]"",
        # }
    """  # noqa: E501

    output_parser: LLMAsAJudgePairwiseOutputParser = LLMAsAJudgePairwiseOutputParser()

    @classmethod
    def from_llm(  # type: ignore[override]
        cls,
        llm: BaseChatModel,
    ) -> PairwiseStringEvalChain:
        """Initialize the LabeledPairwiseStringEvalChain from a ChatModel.

        Args:
            llm (BaseChatModel): The ChatModel to use for evaluation (GPT-4 is recommended).

        Returns:
            LLMAsAJudgePairwiseEvalChain: The initialized LLMAsAJudgePairwiseEvalChain.

        Raises:
            TypeError: If the llm is not a ChatModel.
        """  # noqa: E501
        if not isinstance(llm, BaseChatModel):
            raise TypeError(
                f"This chain requires a ChatModel, but received {type(llm)}."
            )
        if not (isinstance(llm, ChatOpenAI) and llm.model_name.startswith("gpt-4")):
            logger.warning(
                "This chain was only tested with GPT-4. \
                Performance may vary with other models."
            )
        return cls(llm=llm, prompt=COMPARISON_TEMPLATE)
