import os
import re
from typing import List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class EtherscanLoader(BaseLoader):
    """
    Load transactions from an account on Ethereum mainnet.

    The Loader use Etherscan API to interact with Ethereum mainnet.

    ETHERSCAN_API_KEY environment variable must be set use this loader.
    """

    def __init__(
        self,
        account_address: str,
        api_key: str = "docs-demo",
        filter: str = "normal_transaction",
        page: int = 1,
        offset: int = 10,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: str = "desc",
    ):
        self.account_address = account_address
        self.api_key = os.environ.get("ETHERSCAN_API_KEY") or api_key
        self.filter = filter
        self.page = page
        self.offset = offset
        self.start_block = start_block
        self.end_block = end_block
        self.sort = sort

        if not self.api_key:
            raise ValueError("Etherscan API key not provided")

        if not re.match(r"^0x[a-fA-F0-9]{40}$", self.account_address):
            raise ValueError(f"Invalid contract address {self.account_address}")
        if filter not in [
            "normal_transaction",
            "internal_transaction",
            "erc20_transaction",
            "eth_balance",
            "erc721_transaction",
            "erc1155_transaction",
        ]:
            raise ValueError(f"Invalid filter {filter}")

    def load(self) -> List[Document]:
        if self.filter == "normal_transaction":
            return self.getNormTx()
        elif self.filter == "internal_transaction":
            return self.getInternalTx()
        elif self.filter == "erc20_transaction":
            return self.getERC20Tx()
        elif self.filter == "eth_balance":
            return self.getEthBalance()
        elif self.filter == "erc721_transaction":
            return self.getERC721Tx()
        elif self.filter == "erc1155_transaction":
            return self.getERC1155Tx()
        else:
            raise ValueError(f"Invalid filter {filter}")

    def getNormTx(self) -> List[Document]:
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={self.account_address}&startblock=${self.start_block}&endblock=${self.end_block}&page={self.page}&offset={self.offset}&sort={self.sort}&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print("Error occurred while making the request:", e)
        items = response.json()["result"]
        result = []
        if len(items) == 0:
            return [Document(page_content="")]
        for item in items:
            content = str(item)
            metadata = {"from": item["from"], "tx_hash": item["hash"], "to": item["to"]}
            result.append(Document(page_content=content, metadata=metadata))
        print(len(result))
        return result

    def getEthBalance(self) -> List[Document]:
        url = f"https://api.etherscan.io/api?module=account&action=balance&address={self.account_address}&tag=latest&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print("Error occurred while making the request:", e)
        return [Document(page_content=response.json()["result"])]

    def getInternalTx(self) -> List[Document]:
        url = f"https://api.etherscan.io/api?module=account&action=txlistinternal&address={self.account_address}&startblock=${self.start_block}&endblock=${self.end_block}&page={self.page}&offset={self.offset}&sort={self.sort}&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print("Error occurred while making the request:", e)
        items = response.json()["result"]
        result = []
        if len(items) == 0:
            return [Document(page_content="")]
        for item in items:
            content = str(item)
            metadata = {"from": item["from"], "tx_hash": item["hash"], "to": item["to"]}
            result.append(Document(page_content=content, metadata=metadata))
        return result

    def getERC20Tx(self) -> List[Document]:
        url = f"https://api.etherscan.io/api?module=account&action=tokentx&address={self.account_address}&startblock=${self.start_block}&endblock=${self.end_block}&page={self.page}&offset={self.offset}&sort={self.sort}&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print("Error occurred while making the request:", e)
        items = response.json()["result"]
        result = []
        if len(items) == 0:
            return [Document(page_content="")]
        for item in items:
            content = str(item)
            metadata = {"from": item["from"], "tx_hash": item["hash"], "to": item["to"]}
            result.append(Document(page_content=content, metadata=metadata))
        return result

    def getERC721Tx(self) -> List[Document]:
        url = f"https://api.etherscan.io/api?module=account&action=tokennfttx&address={self.account_address}&startblock=${self.start_block}&endblock=${self.end_block}&page={self.page}&offset={self.offset}&sort={self.sort}&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print("Error occurred while making the request:", e)
        items = response.json()["result"]
        result = []
        if len(items) == 0:
            return [Document(page_content="")]
        for item in items:
            content = str(item)
            metadata = {"from": item["from"], "tx_hash": item["hash"], "to": item["to"]}
            result.append(Document(page_content=content, metadata=metadata))
        return result

    def getERC1155Tx(self) -> List[Document]:
        url = f"https://api.etherscan.io/api?module=account&action=token1155tx&address={self.account_address}&startblock=${self.start_block}&endblock=${self.end_block}&page={self.page}&offset={self.offset}&sort={self.sort}&apikey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print("Error occurred while making the request:", e)
        items = response.json()["result"]
        result = []
        if len(items) == 0:
            return [Document(page_content="")]
        for item in items:
            content = str(item)
            metadata = {"from": item["from"], "tx_hash": item["hash"], "to": item["to"]}
            result.append(Document(page_content=content, metadata=metadata))
        return result
