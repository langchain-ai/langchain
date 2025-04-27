import unittest
import sys
from io import StringIO
from contextlib import contextmanager
from langchain.memory.persona import PersonaMemory, EnrichedMessage
from langchain.schema import AIMessage, HumanMessage, BaseMessage
import json
from typing import List, Optional

@contextmanager
def capture_output():
    """Capture stdout and stderr."""
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

class MockChatModel:
    """A simple mock chat model that can simulate both success and failure cases."""
    
    def __init__(self, response: Optional[str] = None, should_fail: bool = False, error_msg: str = ""):
        self.response = response
        self.should_fail = should_fail
        self.error_msg = error_msg
        
    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Simulate chat model invocation."""
        if self.should_fail:
            raise ValueError(self.error_msg)
        return AIMessage(content=self.response)

class TestPersonaMemory(unittest.TestCase):
    def setUp(self):
        """Set up test-level attributes."""
        self.maxDiff = None
        unittest.TestResult.showAll = True

    def _create_successful_chat_model(self):
        """Create a mock chat model that returns successful responses."""
        response = json.dumps({
            "traits": {
                "confident": 2,
                "analytical": 1,
                "professional": 1
            }
        })
        return MockChatModel(response=response)

    def _create_failing_chat_model(self, error_type: str = "rate_limit"):
        """Create a mock chat model that simulates failures."""
        if error_type == "rate_limit":
            error_msg = "Rate limit exceeded"
        elif error_type == "authentication":
            error_msg = "Invalid API key"
        else:
            error_msg = "Internal server error"
        
        return MockChatModel(should_fail=True, error_msg=error_msg)

    def test_enriched_message_creation(self):
        input_data = {
            "id": "msg-001",
            "content": "Hello!"
        }
        message = EnrichedMessage(**input_data)
        self.assertEqual(message.id, "msg-001")
        self.assertEqual(message.content, "Hello!")
        self.assertEqual(message.traits, [])
        self.assertEqual(message.metadata, {})

    def test_persona_memory_initialization(self):
        memory = PersonaMemory()
        self.assertEqual(memory.memory_key, "persona")
        self.assertEqual(memory.input_key, "input")
        self.assertEqual(memory.output_key, "output")
        self.assertEqual(memory.k, 10)
        self.assertEqual(memory.persona_traits, [])
        self.assertEqual(memory.recent_messages, [])

    def test_memory_variables_property(self):
        custom_key = "custom_memory_key"
        memory = PersonaMemory(memory_key=custom_key)
        result = memory.memory_variables
        self.assertEqual(result, [custom_key])

    def test_default_trait_detection_simple_text(self):
        memory = PersonaMemory()
        text = "I'm so sorry again! Hello!"
        traits = memory._detect_traits(text)
        self.assertIn("apologetic", traits)
        self.assertIn("friendly", traits)
        self.assertIn("enthusiastic", traits)
        self.assertEqual(traits["apologetic"], 1)
        self.assertEqual(traits["friendly"], 1)
        self.assertEqual(traits["enthusiastic"], 2)

    def test_default_trait_detection_no_traits(self):
        memory = PersonaMemory()
        text = "This sentence has no emotional clues."
        traits = memory._detect_traits(text)
        self.assertEqual(traits, {})

    def test_fallback_to_default_on_engine_failure(self):
        def broken_detector(text: str):
            raise ValueError("Simulated engine failure")

        memory = PersonaMemory(trait_detection_engine=broken_detector)
        text = "Hey, sorry about that."
        traits = memory._detect_traits(text)
        self.assertIn("apologetic", traits)
        self.assertIn("friendly", traits)
        self.assertEqual(traits["apologetic"], 1)
        self.assertEqual(traits["friendly"], 1)

    def test_external_trait_detection_mock(self):
        def mock_trait_detector(text: str):
            return {"mocked_trait": 2}

        memory = PersonaMemory(trait_detection_engine=mock_trait_detector)
        text = "This is a mock test."
        traits = memory._detect_traits(text)
        self.assertIn("mocked_trait", traits)
        self.assertEqual(traits["mocked_trait"], 2)

    def test_multiple_outputs_accumulate_traits(self):
        memory = PersonaMemory()
        output_texts = [
            "I'm sorry, truly sorry about the confusion.",
            "Hello there! Great job on that task!"
        ]

        for text in output_texts:
            traits = memory._detect_traits(text)
            memory.persona_traits.extend(list(traits.keys()))

        self.assertIn("apologetic", memory.persona_traits)
        self.assertIn("friendly", memory.persona_traits)
        self.assertIn("enthusiastic", memory.persona_traits)
        self.assertEqual(len(memory.persona_traits), 3)

    def test_clear_method_resets_memory(self):
        memory = PersonaMemory()
        initial_traits = ["friendly", "apologetic"]
        initial_messages = ["dummy_message"]
        memory.persona_traits = initial_traits
        memory.recent_messages = initial_messages
        memory.clear()
        self.assertEqual(memory.persona_traits, [])
        self.assertEqual(memory.recent_messages, [])

    def test_successful_api_detection(self):
        chat_model = self._create_successful_chat_model()
        
        def trait_detector(text: str):
            messages = [HumanMessage(content=text)]
            response = chat_model.invoke(messages)
            return json.loads(response.content)["traits"]
            
        memory = PersonaMemory(trait_detection_engine=trait_detector)
        text = "Based on the analysis, I can confidently say this is the best approach."
        traits = memory._detect_traits(text)
        self.assertIn("confident", traits)
        self.assertIn("analytical", traits)
        self.assertIn("professional", traits)
        self.assertEqual(traits["confident"], 2)
        self.assertEqual(traits["analytical"], 1)
        self.assertEqual(traits["professional"], 1)

    def test_rate_limit_failure(self):
        chat_model = self._create_failing_chat_model("rate_limit")
        
        def trait_detector(text: str):
            messages = [HumanMessage(content=text)]
            response = chat_model.invoke(messages)
            return json.loads(response.content)["traits"]
            
        memory = PersonaMemory(trait_detection_engine=trait_detector)
        text = "I'm so sorry about the confusion."
        traits = memory._detect_traits(text)
        self.assertIn("apologetic", traits)  # Should fall back to default detection

    def test_authentication_failure(self):
        chat_model = self._create_failing_chat_model("authentication")
        
        def trait_detector(text: str):
            messages = [HumanMessage(content=text)]
            response = chat_model.invoke(messages)
            return json.loads(response.content)["traits"]
            
        memory = PersonaMemory(trait_detection_engine=trait_detector)
        text = "I apologize for the mistake."
        traits = memory._detect_traits(text)
        self.assertIn("apologetic", traits)  # Should fall back to default detection

    def test_server_error_failure(self):
        chat_model = self._create_failing_chat_model("server_error")
        
        def trait_detector(text: str):
            messages = [HumanMessage(content=text)]
            response = chat_model.invoke(messages)
            return json.loads(response.content)["traits"]
            
        memory = PersonaMemory(trait_detection_engine=trait_detector)
        text = "I'm very sorry about that."
        traits = memory._detect_traits(text)
        self.assertIn("apologetic", traits)  # Should fall back to default detection

if __name__ == "__main__":
    unittest.main()