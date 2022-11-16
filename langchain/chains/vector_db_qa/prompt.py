# flake8: noqa
from langchain.prompts import Prompt

prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If a source is provided for a particular piece of context, please cite that source when repsonding.

For example:
"Answer goes here..." - [source goes here]

{context}

Question: {question}
Helpful Answer:"""
prompt = Prompt(template=prompt_template, input_variables=["context", "question"])
