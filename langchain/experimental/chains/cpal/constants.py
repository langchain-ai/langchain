from enum import Enum


class Constant(Enum):
    """
    avoids hardcoding the same common chain input and output keys
    as a string in multiple places
    """

    narrative_input = "narrative_input"  # human authored LLM input
    chain_answer = "chain_answer"  # Post-processing LLM completion
    chain_data = "chain_data"  # pydantic obj of LLM completion
