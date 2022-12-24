
from __future__ import annotations
from typing import List

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import Counter
from langchain.llms.base import BaseLLM
template = """You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score it as either CORRECT or INCORRECT.
Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here
Please remember to grade them based on being factually accurate. Begin!
QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""
prompt = PromptTemplate(input_variables=['query', 'result', "answer"], template=template)



class QAEvaluator:
    """Class with methods for evaluating question/answering systems."""

    @classmethod
    def from_llm(cls, llm: BaseLLM) -> QAEvaluator:
        """Load base evaluator from an LLM."""
        llm_chain = LLMChain(llm=llm, prompt=prompt, output_key="grade")
        return cls(llm_chain)

    def __init__(self, evaluation_chain: LLMChain):
        """Initialize with the LLM chain used for grading."""
        self.evaluation_chain = evaluation_chain

    def score(self, evaluated_outputs: List[dict]):
        """Score the evaluations."""
        return Counter([output["grade"] for output in evaluated_outputs])

    def run_eval(self, predictions: List[dict]):
        """Run the evaluation chain. Correcting it to all caps, as needed."""
        e = self.evaluation_chain.apply(predictions)
        for i, v in enumerate(e):
            if "Correct" in v:
                e[i] = "CORRECT"
            if "Incorrect" in v:
                e[i] = "INCORRECT"
        return e
    
    def run_and_score(self, examples: List[dict], chain):
        """Run and score predictions."""
        predictions = chain.apply(examples)
        evaluated_outputs = self.run_eval(predictions)
        score = self.score(evaluated_outputs)
        return predictions, evaluated_outputs, score
