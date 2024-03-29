import os
import re
import time
from enum import Enum
from typing import List, Optional

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class BlockchainType(Enum):
    """Enumerator of the supported blockchains."""

    ETH_MAINNET = "eth-mainnet"
    ETH_GOERLI = "eth-goerli"
    POLYGON_MAINNET = "polygon-mainnet"
    POLYGON_MUMBAI = "polygon-mumbai"


class BlockchainDocumentLoader(BaseLoader):
    """Load elements from a blockchain smart contract.

    The supported blockchains are: Ethereum mainnet, Ethereum Goerli testnet,
    Polygon mainnet, and Polygon Mumbai testnet.

    If no BlockchainType is specified, the default is Ethereum mainnet.

    The Loader uses the Alchemy API to interact with the blockchain.
    ALCHEMY_API_KEY environment variable must be set to use this loader.

    The API returns 100 NFTs per request and can be paginated using the
    startToken parameter.

    If get_all_tokens is set to True, the loader will get all tokens
    on the contract.  Note that for contracts with a large number of tokens,
    this may take a long time (e.g. 10k tokens is 100 requests).
    Default value is false for this reason.

    The max_execution_time (sec) can be set to limit the execution time
    of the loader.

    Future versions of this loader can:
        - Support additional Alchemy APIs (e.g. getTransactions, etc.)
        - Support additional blockain APIs (e.g. Infura, Opensea, etc.)
    """

    def __init__(
        self,
        contract_address: str,
        blockchainType: BlockchainType = BlockchainType.ETH_MAINNET,
        api_key: str = "docs-demo",
        startToken: str = "",
        get_all_tokens: bool = False,
        max_execution_time: Optional[int] = None,
    ):
        """

        Args:
            contract_address: The address of the smart contract.
            blockchainType: The blockchain type.
            api_key: The Alchemy API key.
            startToken: The start token for pagination.
            get_all_tokens: Whether to get all tokens on the contract.
            max_execution_time: The maximum execution time (sec).
        """
        self.contract_address = contract_address
        self.blockchainType = blockchainType.value
        self.api_key = os.environ.get("ALCHEMY_API_KEY") or api_key
        self.startToken = startToken
        self.get_all_tokens = get_all_tokens
        self.max_execution_time = max_execution_time

        if not self.api_key:
            raise ValueError("Alchemy API key not provided.")

        if not re.match(r"^0x[a-fA-F0-9]{40}$", self.contract_address):
            raise ValueError(f"Invalid contract address {self.contract_address}")

    def load(self) -> List[Document]:
        result = []

        current_start_token = self.startToken

        start_time = time.time()

        while True:
            url = (
                f"https://{self.blockchainType}.g.alchemy.com/nft/v2/"
                f"{self.api_key}/getNFTsForCollection?withMetadata="
                f"True&contractAddress={self.contract_address}"
                f"&startToken={current_start_token}"
            )

            response = requests.get(url)

            if response.status_code != 200:
                raise ValueError(
                    f"Request failed with status code {response.status_code}"
                )

            items = response.json()["nfts"]

            if not items:
                break

            for item in items:
                content = str(item)
                tokenId = item["id"]["tokenId"]
                metadata = {
                    "source": self.contract_address,
                    "blockchain": self.blockchainType,
                    "tokenId": tokenId,
                }
                result.append(Document(page_content=content, metadata=metadata))

            # exit after the first API call if get_all_tokens is False
            if not self.get_all_tokens:
                break

            # get the start token for the next API call from the last item in array
            current_start_token = self._get_next_tokenId(result[-1].metadata["tokenId"])

            if (
                self.max_execution_time is not None
                and (time.time() - start_time) > self.max_execution_time
            ):
                raise RuntimeError("Execution time exceeded the allowed time limit.")

        if not result:
            raise ValueError(
                f"No NFTs found for contract address {self.contract_address}"
            )

        return result

    # add one to the tokenId, ensuring the correct tokenId format is used
    def _get_next_tokenId(self, tokenId: str) -> str:
        value_type = self._detect_value_type(tokenId)

        if value_type == "hex_0x":
            value_int = int(tokenId, 16)
        elif value_type == "hex_0xbf":
            value_int = int(tokenId[2:], 16)
        else:
            value_int = int(tokenId)

        result = value_int + 1

        if value_type == "hex_0x":
            return "0x" + format(result, "0" + str(len(tokenId) - 2) + "x")
        elif value_type == "hex_0xbf":
            return "0xbf" + format(result, "0" + str(len(tokenId) - 4) + "x")
        else:
            return str(result)

    # A smart contract can use different formats for the tokenId
    @staticmethod
    def _detect_value_type(tokenId: str) -> str:
        if isinstance(tokenId, int):
            return "int"
        elif tokenId.startswith("0x"):
            return "hex_0x"
        elif tokenId.startswith("0xbf"):
            return "hex_0xbf"
        else:
            return "hex_0xbf"
