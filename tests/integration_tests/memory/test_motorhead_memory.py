import unittest
from io import StringIO
from unittest.mock import patch
import json
from langchain.memory.motorhead_memory import MotorheadMemoryMessage, MotorheadMemoryInput, MotorheadMemory

class TestMotorheadMemoryMessage(unittest.TestCase):
    def test_initialization(self):
        message = MotorheadMemoryMessage("AI", "Hello")
        self.assertEqual(message.role, "AI")
        self.assertEqual(message.content, "Hello")

class TestMotorheadMemoryInput(unittest.TestCase):
    def test_initialization(self):
        input_data = MotorheadMemoryInput(None, False, "test_session")
        self.assertIsNone(input_data.chat_history)
        self.assertFalse(input_data.return_messages)
        self.assertEqual(input_data.session_id, "test_session")
        self.assertIsNone(input_data.input_key)
        self.assertIsNone(input_data.output_key)

class TestMotorheadMemory(unittest.TestCase):
    def setUp(self):
        self.mock_response_data = {
            "messages": [
                {"role": "Human", "content": "Hi"},
                {"role": "AI", "content": "Hello there"}
            ],
            "context": "testing"
        }

    @patch('requests.get')
    def test_init(self, mock_get):
        mock_get.return_value.json.return_value = self.mock_response_data
        motorhead_memory = MotorheadMemory(MotorheadMemoryInput(None, False, 'test_session'))

        self.assertEqual(len(motorhead_memory.chat_history.messages), 2)
        self.assertEqual(motorhead_memory.chat_history.messages[0].role, 'user')
        self.assertEqual(motorhead_memory.chat_history.messages[0].content, 'Hi')
        self.assertEqual(motorhead_memory.chat_history.messages[1].role, 'assistant')
        self.assertEqual(motorhead_memory.chat_history.messages[1].content, 'Hello there')
        self.assertEqual(motorhead_memory.context, 'testing')

    @patch('requests.post')
    def test_save_context(self, mock_post):
        motorhead_memory = MotorheadMemory(MotorheadMemoryInput(None, False, 'test_session'))
        motorhead_memory.save_context({"input": "How are you?"}, {"response": "I'm fine, thank you."})

        self.assertTrue(mock_post.called)
        mock_post.assert_called_with(f"{motorhead_memory.motorhead_url}/sessions/{motorhead_memory.session_id}/memory", timeout=motorhead_memory.timeout, json={
            "messages": [
                {"role": "Human", "content": "How are you?"},
                {"role": "AI", "content": "I'm fine, thank you."}
            ]
        }, headers={"Content-Type": "application/json"})

if __name__ == '__main__':
    unittest.main()
