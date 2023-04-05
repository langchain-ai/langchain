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

context_template = """You are a teacher grading a quiz.
You are given a question, the contex the question is about, and the student's answer You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Example Format:
QUESTION: question here
CONTEXT: context the question is about here
STUDENT ANSWER: student's answer here
GRADE: CORRECT or INCORRECT here

Please remember to grade them based on being factually accurate. Begin!

QUESTION: {query}
CONTEXT: {context}
STUDENT ANSWER: {result}
GRADE:"""
CONTEXT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "result"], template=context_template
)


cot_template = """You are a teacher grading a quiz.
You are given a question, the contex the question is about, and the student's answer You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.
Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.

Example Format:
QUESTION: question here
CONTEXT: context the question is about here
STUDENT ANSWER: student's answer here
EXPLANATION: step by step reasoning here
GRADE: CORRECT or INCORRECT here

Please remember to grade them based on being factually accurate. Begin!

QUESTION: {query}
CONTEXT: {context}
STUDENT ANSWER: {result}
EXPLANATION:"""
COT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "result"], template=cot_template
)
