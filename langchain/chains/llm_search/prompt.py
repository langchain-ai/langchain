# flake8: noqa
from langchain.prompts import PromptTemplate

template = """Between >>> and <<< are the raw search result text from google.
Extract the answer to the question '{query}' or say "not found" if the information is not contained.
Use the format
Extracted:<answer or "not found">
>>> {search_results} <<<
Extracted:"""

PROMPT = PromptTemplate(
    input_variables=["query", "search_results"],
    template=template,
)
