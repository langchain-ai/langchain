# flake8: noqa
from langchain.prompts import PromptTemplate
from langchain.utilities.locale import _

prompt_template = _("""Write a concise summary of the following:


"{text}"


CONCISE SUMMARY:""")
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
