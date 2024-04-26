import os

from langchain_upstage import UpstageGroundednessCheck

os.environ["UPSTAGE_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test embedding model initialization."""
    UpstageGroundednessCheck()
    UpstageGroundednessCheck(upstage_api_key="key")
    UpstageGroundednessCheck(api_key="key")
