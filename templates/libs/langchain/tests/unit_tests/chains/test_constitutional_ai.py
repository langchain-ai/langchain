"""Unit tests for the Constitutional AI chain."""
from langchain.chains.constitutional_ai.base import ConstitutionalChain

TEXT_ONE = """ This text is bad.

Revision request: Make it better.

Revision:"""

TEXT_TWO = """ This text is bad.\n\n"""

TEXT_THREE = """ This text is bad.

Revision request: Make it better.

Revision: Better text"""


def test_critique_parsing() -> None:
    """Test parsing of critique text."""
    for text in [TEXT_ONE, TEXT_TWO, TEXT_THREE]:
        critique = ConstitutionalChain._parse_critique(text)

        assert (
            critique.strip() == "This text is bad."
        ), f"Failed on {text} with {critique}"
