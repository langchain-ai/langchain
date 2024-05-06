# flake8: noqa
from langchain_core.prompts import PromptTemplate

template = """You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""
PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"], template=template
)

context_template = """You are a teacher grading a quiz.
You are given a question, the context the question is about, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Example Format:
QUESTION: question here
CONTEXT: context the question is about here
STUDENT ANSWER: student's answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
CONTEXT: {context}
STUDENT ANSWER: {result}
GRADE:"""
CONTEXT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "result"], template=context_template
)


cot_template = """You are a teacher grading a quiz.
You are given a question, the context the question is about, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.
Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.

Example Format:
QUESTION: question here
CONTEXT: context the question is about here
STUDENT ANSWER: student's answer here
EXPLANATION: step by step reasoning here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
CONTEXT: {context}
STUDENT ANSWER: {result}
EXPLANATION:"""
COT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "result"], template=cot_template
)


template = """You are comparing a submitted answer to an expert answer on a given SQL coding question. Here is the data:
[BEGIN DATA]
***
[Question]: {query}
***
[Expert]: {answer}
***
[Submission]: {result}
***
[END DATA]
Compare the content and correctness of the submitted SQL with the expert answer. Ignore any differences in whitespace, style, or output column names. The submitted answer may either be correct or incorrect. Determine which case applies. First, explain in detail the similarities or differences between the expert answer and the submission, ignoring superficial aspects such as whitespace, style or output column names. Do not state the final answer in your initial explanation. Then, respond with either "CORRECT" or "INCORRECT" (without quotes or punctuation) on its own line. This should correspond to whether the submitted SQL and the expert answer are semantically the same or different, respectively. Then, repeat your final answer on a new line."""

SQL_PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"], template=template
)
