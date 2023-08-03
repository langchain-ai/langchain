# flake8: noqa
from langchain.prompts import PromptTemplate

REFINE_PROMPT_TMPL = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary\n"
    "If the context isn't useful, return the original summary."
)
REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=REFINE_PROMPT_TMPL,
)


prompt_template = """Write a concise summary of the following:


"{text}"


CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
