from langchain.chains.vector_db_qa.base import VectorDBQA
from typing import List, Dict, Tuple

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import numpy as np
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

llm = OpenAI(temperature=0)
grader = LLMChain(llm=llm, prompt=prompt, output_key="grade")


def score(examples: List[Dict], chain: VectorDBQA) -> Tuple[List[Dict], float]:
    predictions = chain.apply(examples)
    grade_outputs = grader.apply(predictions)
    score = np.mean([output["grade"] == "CORRECT" for output in grade_outputs])
    return grade_outputs, score