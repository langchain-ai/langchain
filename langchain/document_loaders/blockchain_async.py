import asyncio
import os
import re
from asyncio import Queue
from enum import Enum
from typing import List

import aiohttp

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utilities import asyncio as langchain_asyncio


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
    ALCHEMY_API_KEY environment variable must be set to use this loader.

    The API returns 100 NFTs per request and can be paginated using the
    startToken parameter.

    If get_all_tokens is set to True, the loader will get all tokens
    on the contract.  Note that for contracts with a large number of tokens,
    this may take a long time (e.g. 10k tokens is 100 requests).
    Default value is false for this reason.

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
        max_execution_time: int = 60,
    ):
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
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_load())
        return result

    async def _async_load(self) -> List[Document]:
        result = []

        async with aiohttp.ClientSession() as session:
            if self.get_all_tokens:
                queue: Queue = Queue()
                initial_task = self._fetch_data(session, str(0))
                queue.put_nowait(initial_task)

                while not queue.empty():
                    task = queue.get_nowait()
                    items = await task

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

                    next_start_token = self._get_next_tokenId(
                        result[-1].metadata["tokenId"]
                    )

                    next_task = self._fetch_data(session, next_start_token)
                    queue.put_nowait(next_task)
            else:
                items = await self._fetch_data(session, str(0))

                for item in items:
                    content = str(item)
                    tokenId = item["id"]["tokenId"]
                    metadata = {
                        "source": self.contract_address,
                        "blockchain": self.blockchainType,
                        "tokenId": tokenId,
                    }
                    result.append(Document(page_content=content, metadata=metadata))

        if not result:
            raise ValueError(
                f"No NFTs found for contract address {self.contract_address}"
            )

        return result

    async def _fetch_data(
        self, session: aiohttp.ClientSession, start_token: str
    ) -> List[dict]:
        url = (
            f"https://{self.blockchainType}.g.alchemy.com/nft/v2/"
            f"{self.api_key}/getNFTsForCollection?withMetadata="
            f"True&contractAddress={self.contract_address}"
            f"&startToken={start_token}"
        )

        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"Request failed with status code {response.status}")

            items = (await response.json())["nfts"]
            return items

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
