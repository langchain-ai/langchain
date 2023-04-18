# flake8: noqa
prompt_template = """Given the following question and context, extract any part of the context *as is* that is relevant to answer the question. If none of the context is relevant return {no_output_str}.

> Question: {{question}}
> Context:
>>>
{{context}}
>>>
Extracted relevant parts:"""
