# flake8: noqa
from langchain_core.prompts import PromptTemplate

prompt_template = """Тебе будет дан текст, напиши его краткое содержание:
"{text}"

КРАТКОЕ СОДЕРЖАНИЕ:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
