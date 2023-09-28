"""This is a template for a custom chain.

Edit this file to implement your chain logic.
"""

from typing import Optional

from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import Runnable

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""  # noqa: E501
human_template = "{text}"


def get_chain(model: Optional[BaseLanguageModel] = None) -> Runnable:
    """Return a chain."""
    model = model or ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", human_template),
        ]
    )
    return prompt | model | CommaSeparatedListOutputParser()
