"""Comparison method from "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena" by (Zheng, et al. 2023).

This method achieves 85% agreement with humans when using GPT-4."""  # noqa: E501
from langchain.evaluation.comparison.llm_as_a_judge.eval_chain import (
    LLMAsAJudgePairwiseEvalChain,
)

__all__ = ["LLMAsAJudgePairwiseEvalChain"]
