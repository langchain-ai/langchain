# flake8: noqa
from langchain.prompts import PromptTemplate

REFINE_PROMPT_TMPL = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original summary"
    "If the context isn't useful, return the original summary."
)
REFINE_PROMPT = PromptTemplate(
    input_variables=["query_str", "existing_answer", "context_str"],
    template=DEFAULT_REFINE_PROMPT_TMPL,
)


prompt_template = """Write a concise summary of the following:


{text}


CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

