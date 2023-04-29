import os

import pytest

from langchain.document_loaders import BlockchainDocumentLoader
from langchain.document_loaders.blockchain import BlockchainType

if "ALCHEMY_API_KEY" in os.environ:
    alchemyKeySet = True
    apiKey = os.environ["ALCHEMY_API_KEY"]
else:
    alchemyKeySet = False


@pytest.mark.skipif(not alchemyKeySet, reason="Alchemy API key not provided.")
def test_get_nfts_valid_contract() -> None:
    contract_address = (
        "0x1a92f7381b9f03921564a437210bb9396471050c"  # CoolCats contract address
    )
    result = BlockchainDocumentLoader(contract_address).load()
    assert len(result) > 0, "No NFTs returned"


@pytest.mark.skipif(not alchemyKeySet, reason="Alchemy API key not provided.")
def test_get_nfts_with_pagination() -> None:
    contract_address = (
        "0x1a92f7381b9f03921564a437210bb9396471050c"  # CoolCats contract address
    )
    startToken = "20"

    result = BlockchainDocumentLoader(
        contract_address,
        BlockchainType.ETH_MAINNET,
        api_key=apiKey,
        startToken=startToken,
    ).load()

    assert len(result) > 0, "No NFTs returned"


@pytest.mark.skipif(not alchemyKeySet, reason="Alchemy API key not provided.")
def test_get_nfts_polygon() -> None:
    contract_address = (
        "0x448676ffCd0aDf2D85C1f0565e8dde6924A9A7D9"  # Polygon contract address
    )
    result = BlockchainDocumentLoader(
        contract_address, BlockchainType.POLYGON_MAINNET
    ).load()
    assert len(result) > 0, "No NFTs returned"


@pytest.mark.skipif(not alchemyKeySet, reason="Alchemy API key not provided.")
def test_get_nfts_invalid_contract() -> None:
    contract_address = (
        "0x111D4e82EA7eCA7F62c3fdf6D39A541be95Bf111"  # Invalid contract address
    )

    with pytest.raises(ValueError) as error_NoNfts:
        BlockchainDocumentLoader(contract_address).load()

    assert (
        str(error_NoNfts.value)
        == "No NFTs found for contract address " + contract_address
    )
