"""Utility functions related to question answering."""
from __future__ import annotations
from typing import List

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
import re
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

template = """You are a teacher coming up with questions to ask on a quiz. 
Given the following document, please generate a question and answer based on that document.

Example Format:
<Begin Document>
...
<End Document>
QUESTION: question here
ANSWER: answer here

These questions should be detailed and be based explicitly on information in the document. Begin!

<Begin Document>
{doc}
<End Document>"""
generate_question_prompt = PromptTemplate(input_variables=['doc'], template=template)


def generate_questions(llm: BaseLLM, texts: List[str]) -> List[dict]:
    """Generate a question/answer pair based on each piece of text."""
    llm_chain = LLMChain(llm=llm, prompt=generate_question_prompt)
    outputs = llm_chain.apply([{"doc": d} for d in texts])
    pairs = []
    for output in outputs:
        regex = r"QUESTION: (.*?)\nANSWER: (.*)"
        match = re.search(regex, output['text'])
        if match:
            question = match.group(1)
            answer = match.group(2)
            pairs.append({"query": question, "answer": answer})
    return pairs


def evaluate_question_answering(
        llm: BaseLLM,
        examples: List[dict],
        predictions: List[dict],
        question_key: str = "query",
        answer_key: str = "answer",
        prediction_key: str = "result",
        output_key: str = "grade"
) -> List[dict]:
    """Evaluate question answering examples and predictions."""
    inputs = []
    for i, example in enumerate(examples):
        _input = {
            "query": example[question_key],
            "answer": example[answer_key],
            "result": predictions[i][prediction_key]
        }
        inputs.append(_input)
    evaluation_chain = LLMChain(llm=llm, prompt=prompt, output_key=output_key)
    return evaluation_chain.apply(inputs)

