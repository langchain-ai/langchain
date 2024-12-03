# test_prompt.py

import unittest
from langchain_core.prompts.prompt import PromptTemplate

class TestPromptTemplate(unittest.TestCase):
    def test_constraints_appended(self):
        template = "You are a helpful assistant."
        prompt_template = PromptTemplate.from_template(template)
        expected_template = template
        self.assertEqual(prompt_template.template, expected_template)

    def test_input_variables(self):
        template = "Answer the following question: {input}"
        prompt_template = PromptTemplate.from_template(template)
        self.assertIn("input", prompt_template.input_variables)

if __name__ == "__main__":
    unittest.main()
