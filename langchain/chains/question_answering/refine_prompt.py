# flake8: noqa
from langchain.prompts import PromptTemplate

DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
DEFAULT_REFINE_PROMPT = PromptTemplate(
    input_variables=["query_str", "existing_answer", "context_msg"],
    template=DEFAULT_REFINE_PROMPT_TMPL,
)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    input_variables=["context_str", "query_str"], template=DEFAULT_TEXT_QA_PROMPT_TMPL
)
