
import pytest
from langchain_nanoempire import NanoEmpireClient

@pytest.mark.asyncio
async def test_client_init():
    client = NanoEmpireClient()
    assert client is not None
