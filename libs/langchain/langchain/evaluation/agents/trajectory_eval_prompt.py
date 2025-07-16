"""Prompt for trajectory evaluation chain."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

EVAL_TEMPLATE = """An AI language model has been given access to the following set of tools to help answer a user's question.

The tools given to the AI model are:
[TOOL_DESCRIPTIONS]
{tool_descriptions}
[END_TOOL_DESCRIPTIONS]

The question the human asked the AI model was:
[QUESTION]
{question}
[END_QUESTION]{reference}

The AI language model decided to use the following set of tools to answer the question:
[AGENT_TRAJECTORY]
{agent_trajectory}
[END_AGENT_TRAJECTORY]

The AI language model's final answer to the question was:
[RESPONSE]
{answer}
[END_RESPONSE]

Let's to do a detailed evaluation of the AI language model's answer step by step.

We consider the following criteria before giving a score from 1 to 5:

i. Is the final answer helpful?
ii. Does the AI language use a logical sequence of tools to answer the question?
iii. Does the AI language model use the tools in a helpful way?
iv. Does the AI language model use too many steps to answer the question?
v. Are the appropriate tools used to answer the question?"""  # noqa: E501

EXAMPLE_INPUT = """An AI language model has been given access to the following set of tools to help answer a user's question.

The tools given to the AI model are:
[TOOL_DESCRIPTIONS]
Tool 1:
Name: Search
Description: useful for when you need to ask with search

Tool 2:
Name: Lookup
Description: useful for when you need to ask with lookup

Tool 3:
Name: Calculator
Description: useful for doing calculations

Tool 4:
Name: Search the Web (SerpAPI)
Description: useful for when you need to answer questions about current events
[END_TOOL_DESCRIPTIONS]

The question the human asked the AI model was: If laid the Statue of Liberty end to end, how many times would it stretch across the United States?

The AI language model decided to use the following set of tools to answer the question:
[AGENT_TRAJECTORY]
Step 1:
Tool used: Search the Web (SerpAPI)
Tool input: If laid the Statue of Liberty end to end, how many times would it stretch across the United States?
Tool output: The Statue of Liberty was given to the United States by France, as a symbol of the two countries' friendship. It was erected atop an American-designed ...
[END_AGENT_TRAJECTORY]

[RESPONSE]
The AI language model's final answer to the question was: There are different ways to measure the length of the United States, but if we use the distance between the Statue of Liberty and the westernmost point of the contiguous United States (Cape Alava, Washington), which is approximately 2,857 miles (4,596 km), and assume that the Statue of Liberty is 305 feet (93 meters) tall, then the statue would stretch across the United States approximately 17.5 times if laid end to end.
[END_RESPONSE]

Let's to do a detailed evaluation of the AI language model's answer step by step.

We consider the following criteria before giving a score from 1 to 5:

i. Is the final answer helpful?
ii. Does the AI language use a logical sequence of tools to answer the question?
iii. Does the AI language model use the tools in a helpful way?
iv. Does the AI language model use too many steps to answer the question?
v. Are the appropriate tools used to answer the question?"""  # noqa: E501

EXAMPLE_OUTPUT = """First, let's evaluate the final answer. The final uses good reasoning but is wrong. 2,857 divided by 305 is not 17.5.\
The model should have used the calculator to figure this out. Second does the model use a logical sequence of tools to answer the question?\
The way model uses the search is not helpful. The model should have used the search tool to figure the width of the US or the height of the statue.\
The model didn't use the calculator tool and gave an incorrect answer. The search API should be used for current events or specific questions.\
The tools were not used in a helpful way. The model did not use too many steps to answer the question.\
The model did not use the appropriate tools to answer the question.\

Judgment: Given the good reasoning in the final answer but otherwise poor performance, we give the model a score of 2.

Score: 2"""  # noqa: E501

EVAL_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessage(
            content="You are a helpful assistant that evaluates language models."
        ),
        HumanMessage(content=EXAMPLE_INPUT),
        AIMessage(content=EXAMPLE_OUTPUT),
        HumanMessagePromptTemplate.from_template(EVAL_TEMPLATE),
    ]
)


TOOL_FREE_EVAL_TEMPLATE = """An AI language model has been given access to a set of tools to help answer a user's question.

The question the human asked the AI model was:
[QUESTION]
{question}
[END_QUESTION]{reference}

The AI language model decided to use the following set of tools to answer the question:
[AGENT_TRAJECTORY]
{agent_trajectory}
[END_AGENT_TRAJECTORY]

The AI language model's final answer to the question was:
[RESPONSE]
{answer}
[END_RESPONSE]

Let's to do a detailed evaluation of the AI language model's answer step by step.

We consider the following criteria before giving a score from 1 to 5:

i. Is the final answer helpful?
ii. Does the AI language use a logical sequence of tools to answer the question?
iii. Does the AI language model use the tools in a helpful way?
iv. Does the AI language model use too many steps to answer the question?
v. Are the appropriate tools used to answer the question?"""  # noqa: E501


TOOL_FREE_EVAL_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessage(
            content="You are a helpful assistant that evaluates language models."
        ),
        HumanMessage(content=EXAMPLE_INPUT),
        AIMessage(content=EXAMPLE_OUTPUT),
        HumanMessagePromptTemplate.from_template(TOOL_FREE_EVAL_TEMPLATE),
    ]
)
