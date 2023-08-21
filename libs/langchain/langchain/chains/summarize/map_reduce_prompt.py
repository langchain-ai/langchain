# flake8: noqa
from langchain.prompts import PromptTemplate

prompt_template = """Напиши краткое резюме следующего:


"{text}"


КРАТКОЕ РЕЗЮМЕ:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
