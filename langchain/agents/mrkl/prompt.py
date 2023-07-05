# flake8: noqa
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """
Structure your response with the following format:

Question: ```the input question you must answer```
Thought: ```you should always think about what to do to answer the input question```
Action: ```the tool you select to use, which should be one of the following: {tool_names}```
Action Input: ```the input to the tool you selected```
Observation: ```the result of the action, this will be computed and given back to you based on your action input```

Repeat the Thought/Action/Action Input/Observation cycle until you find an answer to the Question. Then, when you have the final answer to the question, use the following format:

Thought: ```I now know the final answer```
Final Answer: ```your final answer to the original input question```
"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""
