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

CHAT_INSTRUCTIONS = """You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score it as either CORRECT or INCORRECT.
The format of your response should be `GRADE: ${grade}` with ${grade} being either CORRECT or INCORRECT and nothing more."""

CHAT_RESPONSE_TEMPLATE = """QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}"""

CHAT_COMPARISON_INSTRUCTIONS = """You are a teacher grading a quiz.
You are given a question, the correct answer, Student A's answer and then Student B's answer.
Please describe how Student A's answer compares to Student B's.
Describe them with a comma separated list of adjectives. Example adjectives may include: more verbose, less correct, more succint, etc."""

CHAT_COMPARISON_RESPONSE_TEMPLATE = """QUESTION: {query}
TRUE ANSWER: {answer}
STUDENT A ANSWER: {student_a}
STUDENT B ANSWER: {student_b}"""
