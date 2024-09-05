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

    def test_call_integration(self):
        mock_response = Mock()
        mock_response.question.answers = [Mock(text="Mocked answer")]
        self.mock_client.copilot.ask_question.return_value = mock_response

        result = self.llm._call("What is AI?")
        self.assertEqual(result, "Mocked answer")

    def test_generate_integration(self):
        mock_response = Mock()
        mock_response.question.answers = [Mock(text="Mocked answer")]
        self.mock_client.copilot.ask_question.return_value = mock_response

        result = self.llm._generate(["What is AI?"])
        self.assertEqual(result.generations[0][0].text, "Mocked answer")

    def test_set_model_integration(self):
        supported_models = self.llm.get_supported_models()
        self.assertIn("GPT-4 Chat Model", supported_models)

        self.llm.set_model("GPT-4 Chat Model")
        self.assertEqual(self.llm.model, "GPT-4 Chat Model")
        self.assertEqual(self.mock_client.model_name, "GPT-4 Chat Model")
