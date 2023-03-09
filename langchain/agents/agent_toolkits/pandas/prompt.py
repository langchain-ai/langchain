# flake8: noqa

PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:"""

SUFFIX = """
This is the result of `print(df.head())`:
{df}

Begin!
Question: {input}
{agent_scratchpad}"""
