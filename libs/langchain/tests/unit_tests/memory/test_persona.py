import unittest
import sys
from io import StringIO
from contextlib import contextmanager
from langchain.memory.persona import PersonaMemory, EnrichedMessage
from langchain.schema import AIMessage, HumanMessage, BaseMessage
import json
from typing import List, Optional


class MockChatModel:
    """A simple mock chat model that can simulate both success and failure cases."""

    def __init__(
        self,
        response: Optional[str] = None,
        should_fail: bool = False,
        error_msg: str = "",
    ):
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
        response = json.dumps(
            {"traits": {"confident": 2, "analytical": 1, "professional": 1}}
        )
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
        input_data = {"id": "msg-001", "content": "Hello!"}
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
            "Hello there! Great job on that task!",
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

    def test_save_context_updates_memory(self):
        memory = PersonaMemory()
        input_text = "Hello!"
        output_text = "I'm sorry for the confusion."

        memory.save_context({"input": input_text}, {"output": output_text})
        self.assertIn("apologetic", memory.persona_traits)
        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(memory.recent_messages[0].content, output_text)
        self.assertIn("apologetic", memory.recent_messages[0].traits)

    def test_load_memory_variables_returns_traits(self):
        memory = PersonaMemory()
        memory.persona_traits = ["friendly", "apologetic"]

        result = memory.load_memory_variables({})
        self.assertEqual(result["persona"], {"traits": ["friendly", "apologetic"]})

    def test_save_context_updates_memory_with_multiple_traits(self):
        memory = PersonaMemory()
        input_text = "Hello!"
        output_text = "I'm sorry for the confusion. I'm also friendly and analytical."

        memory.save_context({"input": input_text}, {"output": output_text})
        self.assertIn("apologetic", memory.persona_traits)
        self.assertIn("friendly", memory.persona_traits)
        self.assertIn("analytical", memory.persona_traits)
        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(memory.recent_messages[0].content, output_text)
        self.assertIn("apologetic", memory.recent_messages[0].traits)
        self.assertIn("friendly", memory.recent_messages[0].traits)
        self.assertIn("analytical", memory.recent_messages[0].traits)

    def test_memory_trims_to_k_messages(self):
        memory = PersonaMemory(k=2)

        memory.save_context({"input": "Hi"}, {"output": "Sorry about that!"})
        memory.save_context({"input": "Hello"}, {"output": "Apologies again!"})
        memory.save_context({"input": "Hey"}, {"output": "Great job!"})  # Third message

        self.assertEqual(len(memory.recent_messages), 2)
        self.assertEqual(memory.recent_messages[0].content, "Apologies again!")
        self.assertEqual(memory.recent_messages[1].content, "Great job!")

    def test_load_memory_variables_with_messages(self):
        memory = PersonaMemory()
        output_text = "Sorry about the mistake!"

        memory.save_context({"input": "Hello"}, {"output": output_text})

        result = memory.load_memory_variables({}, include_messages=True)
        self.assertIn("traits", result["persona"])
        self.assertIn("recent_messages", result["persona"])
        self.assertEqual(
            result["persona"]["recent_messages"][0]["content"], output_text
        )

    def test_save_context_with_missing_output(self):
        memory = PersonaMemory()

        memory.save_context({"input": "Hi"}, {})  # No output provided

        self.assertEqual(len(memory.recent_messages), 1)
        self.assertEqual(
            memory.recent_messages[0].content, ""
        )  # Should be empty string
        self.assertEqual(memory.recent_messages[0].traits, [])  # No traits

    def test_double_save_context_creates_two_entries(self):
        memory = PersonaMemory()

        output_text = "Sorry for that mistake."
        memory.save_context({"input": "Hi"}, {"output": output_text})
        memory.save_context(
            {"input": "Hey"}, {"output": output_text}
        )  # Same output again

        self.assertEqual(len(memory.recent_messages), 2)
        self.assertEqual(memory.recent_messages[0].content, output_text)
        self.assertEqual(memory.recent_messages[1].content, output_text)

    def test_full_conversation_simulation_with_failures(self):
        """Simulate a longer conversation with mixed success and fallback during trait detection."""
        memory = PersonaMemory()

        conversation = [
            ("Hi there!", "I'm so happy to meet you!"),  # Enthusiastic
            (
                "Could you help me?",
                "Of course! Happy to assist.",
            ),  # Friendly + Enthusiastic
            (
                "I'm unsure about this plan...",
                "Maybe we should rethink it.",
            ),  # Hesitant + Cautious (simulated failure here)
            (
                "Sorry about the delay.",
                "Apologies! It won't happen again.",
            ),  # Apologetic
        ]

        print("\n\n--- Starting full conversation simulation ---\n")

        # Simulate that at message 3, external engine fails (mock failure)
        def dynamic_trait_detector(text: str):
            if "rethink" in text:
                print("Simulated API Failure during trait detection.\n")
                raise ValueError("Simulated API failure during cautious response")
            print(f"Simulated API Success - analyzing text: '{text}'\n")
            if "happy to meet you" in text:
                return {"enthusiastic": 1, "friendly": 1, "mocked_positive": 1}
            elif "Happy to assist" in text:
                return {"friendly": 1, "enthusiastic": 1, "mocked_positive": 1}
            elif "Apologies" in text:
                return {"apologetic": 1, "mocked_positive": 1}
            return {"mocked_positive": 1}

        memory.trait_detection_engine = dynamic_trait_detector

        for idx, (user_input, agent_output) in enumerate(conversation):
            print(f"--- Message {idx+1} ---")
            print(f"Input: {user_input}")
            print(f"Output: {agent_output}\n")
            memory.save_context({"input": user_input}, {"output": agent_output})

            last_message = memory.recent_messages[-1]
            print(f"Saved Message Traits: {last_message.traits}\n")

        print("--- Final accumulated traits ---")
        print(memory.persona_traits)
        print("\n--- End of conversation simulation ---\n")

        # Assertions:
        # There should be 4 recent messages (conversation length)
        self.assertEqual(len(memory.recent_messages), 4)

        # Traits should have accumulated normally, fallback triggered once
        detected_traits = set(memory.persona_traits)

        expected_traits = {
            "enthusiastic",
            "friendly",
            "hesitant",
            "cautious",
            "apologetic",
            "mocked_positive",
        }

        print(f"Detected traits: {detected_traits}")
        print(f"Expected traits: {expected_traits}")

        for trait in expected_traits:
            self.assertIn(trait, detected_traits)

        # Messages should have traits assigned
        for message in memory.recent_messages:
            self.assertIsInstance(message.traits, list)
            self.assertGreaterEqual(len(message.traits), 1)


if __name__ == "__main__":
    unittest.main()


"""
Just the convo simulation:
python3 -m unittest -v libs.langchain.tests.unit_tests.memory.test_persona.TestPersonaMemory.test_full_conversation_simulation_with_failure

All tests:
python3 -m unittest -v libs.langchain.tests.unit_tests.memory.test_persona

or quitetly:
python3 -m unittest -q libs.langchain.tests.unit_tests.memory.test_persona

"""
