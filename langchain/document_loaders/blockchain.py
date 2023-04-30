import os
import re
from enum import Enum
from typing import List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class BlockchainType(Enum):
    ETH_MAINNET = "eth-mainnet"
    ETH_GOERLI = "eth-goerli"
    POLYGON_MAINNET = "polygon-mainnet"
    POLYGON_MUMBAI = "polygon-mumbai"


class BlockchainDocumentLoader(BaseLoader):
    """Loads elements from a blockchain smart contract into Langchain documents.

    The supported blockchains are: Ethereum mainnet, Ethereum Goerli testnet,
    Polygon mainnet, and Polygon Mumbai testnet.

    If no BlockchainType is specified, the default is Ethereum mainnet.

    The Loader uses the Alchemy API to interact with the blockchain.

    The API returns 100 NFTs per request and can be paginated using the
    startToken parameter.

    ALCHEMY_API_KEY environment variable must be set to use this loader.

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
    ):
        self.contract_address = contract_address
        self.blockchainType = blockchainType.value
        self.api_key = os.environ.get("ALCHEMY_API_KEY") or api_key
        self.startToken = startToken

        if not self.api_key:
            raise ValueError("Alchemy API key not provided.")

        if not re.match(r"^0x[a-fA-F0-9]{40}$", self.contract_address):
            raise ValueError(f"Invalid contract address {self.contract_address}")

    def load(self) -> List[Document]:
        url = (
            f"https://{self.blockchainType}.g.alchemy.com/nft/v2/"
            f"{self.api_key}/getNFTsForCollection?withMetadata="
            f"True&contractAddress={self.contract_address}"
            f"&startToken={self.startToken}"
        )

        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}")

        items = response.json()["nfts"]

        if not (items):
            raise ValueError(
                f"No NFTs found for contract address {self.contract_address}"
            )

        result = []

        for item in items:
            content = str(item)
            tokenId = item["id"]["tokenId"]
            metadata = {
                "source": self.contract_address,
                "blockchain": self.blockchainType,
                "tokenId": tokenId,
            }
            result.append(Document(page_content=content, metadata=metadata))
        return result
