# flake8: noqa
prompt_template = """Given the following question and context, return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
Now respond and ONLY answer YES or NO since you are only able to say 1 word responses.
> Relevant (YES / NO):"""
