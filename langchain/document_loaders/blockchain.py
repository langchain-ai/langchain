import os
import requests
import json

from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

ALCHEMY_API_KEY = os.environ.get("ALCHEMY_API_KEY", "docs-demo")

class BlockchainDocumentLoader(BaseLoader):
    """Loads elements from a blockchain smart contract into Langchain documents.

    For now, the only supported blockchain is Ethereum Mainnet.
    Currently Loader returns all NFTs from a given contract address.
    
    The Loader uses the Alchemy API to interact with the blockchain.
    
    ALCHEMY_API_KEY environment variable must be set to use this loader.
    
    Future versions of this loader can:
        - Support other blockchains (Ethereum testnets, Polygon, etc.)
        - Support additional Alchemy APIs (e.g. getTransactions, etc.)
    """
    
    def __init__(self, contract_address: str, api_key: str = ALCHEMY_API_KEY):
        self.contract_address = contract_address
        self.api_key = api_key
    
    def load(self) -> List[Document]:
            if not self.api_key:
                raise ValueError("Alchemy API key not provided.")
            
            url = f"https://eth-mainnet.g.alchemy.com/nft/v2/{self.api_key}/getNFTsForCollection?withMetadata=True&contractAddress={self.contract_address}"

            response = requests.get(url)

            if response.status_code != 200:
                raise ValueError(f"Request failed with status code {response.status_code}")

            items = response.json()["nfts"]
                      
            result = []
      
            for item in items:
                content = str(item)
                tokenId = item["id"]["tokenId"]
                metadata = {"tokenId": tokenId}
                result.append(Document(page_content=content, metadata=metadata))
            return result
    
