# flake8: noqa
from langchain.utilities.locale import _

prompt_template = _("""Given the following question and context, return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):""")
