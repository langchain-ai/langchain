# flake8: noqa
from langchain.prompts import PromptTemplate

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
PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)
