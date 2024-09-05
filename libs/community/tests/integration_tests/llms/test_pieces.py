import unittest
from unittest.mock import Mock, create_autospec
from pieces_os_client.wrapper import PiecesClient
from langchain_community.llms.pieces import PiecesOSLLM

class TestPiecesOSLLMIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_client = create_autospec(PiecesClient, instance=True)
        self.mock_client.available_models_names = [
            "GPT_3.5", "GPT_4", "T5", "LLAMA_2_7B", "LLAMA_2_13B",
            "GPT-4 Chat Model", "GPT-3.5 Chat Model"
        ]
        self.llm = PiecesOSLLM(client=self.mock_client)

