from enum import Enum
from typing import Type
from langchain.concise import config

from langchain.concise.pattern import pattern
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers.choice import ChoiceOutputParser
from langchain.output_parsers.enum import EnumOutputParser
from langchain.schema import OutputParserException


def choice(
    input: str,
    query: str = None,
    options: list[str] | Type[Enum] = [],
    examples: list[tuple[str, str]] = [],
    llm: BaseLanguageModel = None,
    default=None,
) -> bool:
    """Choose an option from a list of options based on the item and query.

    Args:
        item (str): The item that the choice is being made about.
        query (str): The query that determines what to choose.
        options (list[str]): The list of options to choose from.
        examples (list[str] | tuple[str, str], optional): A list of examples. Items are tuples representing either (item, result) or (query, item, result). If no query is supplied, the top-level query is used. Defaults to [].
        llm (BaseLanguageModel, optional): Language model to override the default LLM. Defaults to config.get_default_llm().

    Returns:
        str: The chosen option.
    """
    llm = llm or config.get_default_model()
    if isinstance(options, type(Enum)):
        parser = EnumOutputParser.from_enum(options, min_distance=1)
    elif isinstance(options, list) and all(isinstance(option, str) for option in options):
        parser = ChoiceOutputParser(options=options, min_distance=1)
    else:
        raise ValueError(
            f"options must be a list of strings or an enum, not {type(options)}"
        )

    try:
        return pattern(
            input=input,
            query=query,
            pattern_name="choice",
            parser=parser,
            examples=examples,
            llm=llm,
        )
    except (ValueError, OutputParserException) as e:
        if default is not None:
            return default
        else:
            print(e)
            raise e
