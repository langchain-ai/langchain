import os
import requests
import json

from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

ALCHEMY_API_KEY = os.environ.get("ALCHEMY_API_KEY", "docs-demo")

class BlockchainDocumentLoader(BaseLoader):
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

            nfts = response.json()["nfts"]
                      
            result = []
      
            for nft in nfts:
                nftContent = str(nft)
                tokenId = nft["id"]["tokenId"]
                metadata = {"tokenId": tokenId}
                result.append(Document(page_content=nftContent, metadata=metadata))
            return result
    
