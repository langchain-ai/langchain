"""
Unit tests for LangChain ZTDS Callback Handler
Validates 4 Core Protocol Invariants (IETF draft-sibiryakov-ztds-protocol-02)
https://datatracker.ietf.org/doc/draft-sibiryakov-ztds-protocol/
"""

import unittest
import uuid
from langchain_core.callbacks.ztds import ZTDSSanitizingCallbackHandler


class MockGeneration:
    def __init__(self, text: str):
        self.text = text


class MockLLMResult:
    def __init__(self, text: str):
        self.generations = [[MockGeneration(text)]]


class TestLangChainZTDSCallback(unittest.TestCase):
    def setUp(self):
        self.handler = ZTDSSanitizingCallbackHandler()

    def test_prompt_sanitization_and_response_restoration(self):
        run_id = uuid.uuid4()
        mock_secret = "".join(["ghp_", "abcdef1234567890", "abcdef1234567890"])
        raw_prompt = f"Query for user admin@corporate.net using token {mock_secret}"
        prompts = [raw_prompt]

        # 1. on_llm_start in-place sanitization
        self.handler.on_llm_start(serialized={}, prompts=prompts, run_id=run_id)
        self.assertNotIn("admin@corporate.net", prompts[0])
        self.assertNotIn(mock_secret, prompts[0])
        self.assertIn("[EMAIL_TOKEN_1]", prompts[0])
        self.assertIn("[API_SECRET_TOKEN_1]", prompts[0])

        # 2. on_llm_end in-place restoration
        mock_result = MockLLMResult("Processed credentials for [EMAIL_TOKEN_1] successfully.")
        self.handler.on_llm_end(mock_result, run_id=run_id)
        final_output = mock_result.generations[0][0].text
        self.assertIn("admin@corporate.net", final_output)
        self.assertNotIn("[EMAIL_TOKEN_1]", final_output)

        # 3. Invariant 3: RAM zeroization
        self.assertNotIn(str(run_id), self.handler._run_maps)
        self.assertNotIn(str(run_id), self.handler._entity_maps)


if __name__ == "__main__":
    unittest.main()
