from langchain_core.utils.function_calling import convert_to_openai_function

from langchain.chains.openai_functions.base import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chains.openai_functions.citation_fuzzy_match import (
    create_citation_fuzzy_match_chain,
    create_citation_fuzzy_match_runnable,
)
from langchain.chains.openai_functions.extraction import (
    create_extraction_chain,
    create_extraction_chain_pydantic,
)
from langchain.chains.openai_functions.qa_with_structure import (
    create_qa_with_sources_chain,
    create_qa_with_structure_chain,
)
from langchain.chains.openai_functions.tagging import (
    create_tagging_chain,
    create_tagging_chain_pydantic,
)
from langchain.chains.structured_output.base import (
    create_openai_fn_runnable,
    create_structured_output_runnable,
    get_openai_output_parser,
)

__all__ = [
    "convert_to_openai_function",
    "create_citation_fuzzy_match_chain",
    "create_citation_fuzzy_match_runnable",
    "create_extraction_chain",
    "create_extraction_chain_pydantic",
    "create_openai_fn_chain",
    "create_openai_fn_runnable",  # backwards compatibility
    "create_qa_with_sources_chain",
    "create_qa_with_structure_chain",
    "create_structured_output_chain",
    "create_structured_output_runnable",  # backwards compatibility
    "create_tagging_chain",
    "create_tagging_chain_pydantic",
    "get_openai_output_parser",  # backwards compatibility
]
