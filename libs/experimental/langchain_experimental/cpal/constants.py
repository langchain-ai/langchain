from enum import Enum


class Constant(Enum):
    """Enum for constants used in the CPAL."""

    narrative_input = "narrative_input"
    chain_answer = "chain_answer"  # natural language answer
    chain_data = "chain_data"  # pydantic instance
