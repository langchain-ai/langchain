from langchain.concise.pattern import pattern
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers.choice import ChoiceOutputParser
from langchain.schema import BaseLanguageModel


def choice(
    input: str,
    query: str = None,
    options: list[str] = [],
    examples: list[tuple[str, str]] = [],
    llm: BaseLanguageModel = None,
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
    return pattern(
        input=input,
        query=query,
        pattern_name="choice",
        parser=ChoiceOutputParser(options=options, min_distance=1),
        examples=examples,
        llm=llm,
    )
