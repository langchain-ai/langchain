import os

from langchain_upstage import GroundednessCheck

os.environ["UPSTAGE_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test embedding model initialization."""
    GroundednessCheck()
    GroundednessCheck(upstage_api_key="key")
    GroundednessCheck(api_key="key")
