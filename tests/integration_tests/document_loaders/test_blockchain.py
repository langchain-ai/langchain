import os
import unittest
from langchain.document_loaders import BlockchainDocumentLoader

class TestBlockchainDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.contract_address = "0x1a92f7381b9f03921564a437210bb9396471050c"  #CoolCats contract address
        self.loader = BlockchainDocumentLoader(self.contract_address)

    def test_get_nfts(self):
        result = self.loader.load()       
        self.assertIsInstance(result, list)

if __name__ == "__main__":
    unittest.main()
