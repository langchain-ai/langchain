import re


def docset_name_to_retriever_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a retriever tool function name.

    Retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "search_".
    2. The rest of the name should be a lowercased string, with underscores for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_retriever_tool_function_name('Earnings Calls')
    'search_earnings_calls'
    >>> docset_name_to_retriever_tool_function_name('COVID-19   Statistics')
    'search_covid_19_statistics'
    >>> docset_name_to_retriever_tool_function_name('2023 Market Report!!!')
    'search_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"search_{name}"


ASSISTANT_SYSTEM_MESSAGE = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""
