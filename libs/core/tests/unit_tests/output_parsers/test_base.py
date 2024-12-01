# test_base.py

import unittest
import logging
from langchain_core.output_parsers.base import BaseOutputParser

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestBaseOutputParser(unittest.TestCase):
    def setUp(self):
        # Initialize the parser with a small history size for testing
        self.parser = BaseOutputParser(history_size=3)

    def test_repetition_detection(self):
        # First input should not detect repetition
        is_repetitive = self.parser.update_and_check_repetition("Thought: I need to calculate 2+2.")
        self.assertFalse(is_repetitive)

        # Second identical input should not yet detect repetition
        is_repetitive = self.parser.update_and_check_repetition("Thought: I need to calculate 2+2.")
        self.assertFalse(is_repetitive)

        # Third identical input should detect repetition
        is_repetitive = self.parser.update_and_check_repetition("Thought: I need to calculate 2+2.")
        self.assertTrue(is_repetitive)

    def test_repetition_with_different_inputs(self):
        # Input different texts
        self.parser.update_and_check_repetition("Thought: I need to calculate 2+2.")
        self.parser.update_and_check_repetition("Thought: I will use the calculator.")
        is_repetitive = self.parser.update_and_check_repetition("Thought: I need to search for information.")

        # Should not detect repetition
        self.assertFalse(is_repetitive)

    def test_history_size_limit(self):
        # Add more entries than history_size
        self.parser.update_and_check_repetition("Thought: Step 1.")
        self.parser.update_and_check_repetition("Thought: Step 2.")
        self.parser.update_and_check_repetition("Thought: Step 3.")
        self.parser.update_and_check_repetition("Thought: Step 4.")  # This should push out "Step 1"

        # Now, repeating "Step 2" twice should not immediately detect repetition
        self.parser.update_and_check_repetition("Thought: Step 2.")
        is_repetitive = self.parser.update_and_check_repetition("Thought: Step 2.")
        self.assertFalse(is_repetitive)

        # Third time should detect repetition
        is_repetitive = self.parser.update_and_check_repetition("Thought: Step 2.")
        self.assertTrue(is_repetitive)

if __name__ == "__main__":
    unittest.main()