import os

import pytest

from langchain.document_loaders import EtherscanLoader

if "ETHERSCAN_API_KEY" in os.environ:
    etherscan_key_set = True
    api_key = os.environ["ETHERSCAN_API_KEY"]
else:
    etherscan_key_set = False


@pytest.mark.skipif(not etherscan_key_set, reason="Etherscan API key not provided.")
def test_get_normal_transaction() -> None:
    account_address = "0x9dd134d14d1e65f84b706d6f205cd5b1cd03a46b"
    loader = EtherscanLoader(account_address)
    result = loader.load()
    assert len(result) > 0, "No transactions returned"


@pytest.mark.skipif(not etherscan_key_set, reason="Etherscan API key not provided.")
def test_get_internal_transaction() -> None:
    account_address = "0x9dd134d14d1e65f84b706d6f205cd5b1cd03a46b"
    loader = EtherscanLoader(account_address, filter="internal_transaction")
    result = loader.load()
    assert len(result) > 0, "No transactions returned"


@pytest.mark.skipif(not etherscan_key_set, reason="Etherscan API key not provided.")
def test_get_erc20_transaction() -> None:
    account_address = "0x9dd134d14d1e65f84b706d6f205cd5b1cd03a46b"
    loader = EtherscanLoader(account_address, filter="erc20_transaction")
    result = loader.load()
    assert len(result) > 0, "No transactions returned"


@pytest.mark.skipif(not etherscan_key_set, reason="Etherscan API key not provided.")
def test_get_erc721_transaction() -> None:
    account_address = "0x9dd134d14d1e65f84b706d6f205cd5b1cd03a46b"
    loader = EtherscanLoader(account_address, filter="erc721_transaction")
    result = loader.load()
    assert len(result) > 0, "No transactions returned"


@pytest.mark.skipif(not etherscan_key_set, reason="Etherscan API key not provided.")
def test_get_erc1155_transaction() -> None:
    account_address = "0x9dd134d14d1e65f84b706d6f205cd5b1cd03a46b"
    loader = EtherscanLoader(account_address, filter="erc1155_transaction")
    result = loader.load()
    assert len(result) == 1, "Wrong transactions returned"
    assert result[0].page_content == "", "Wrong transactions returned"


@pytest.mark.skipif(not etherscan_key_set, reason="Etherscan API key not provided.")
def test_get_eth_balance() -> None:
    account_address = "0x9dd134d14d1e65f84b706d6f205cd5b1cd03a46b"
    loader = EtherscanLoader(account_address, filter="eth_balance")
    result = loader.load()
    assert len(result) > 0, "No transactions returned"


@pytest.mark.skipif(not etherscan_key_set, reason="Etherscan API key not provided.")
def test_invalid_filter() -> None:
    account_address = "0x9dd134d14d1e65f84b706d6f205cd5b1cd03a46b"
    with pytest.raises(ValueError) as error_invalid_filter:
        EtherscanLoader(account_address, filter="internal_saction")
    assert str(error_invalid_filter.value) == "Invalid filter internal_saction"
