# test_prompt.py

import unittest
from langchain_core.prompts.prompt import PromptTemplate

class TestPromptTemplate(unittest.TestCase):
    def test_constraints_appended(self):
        template = "You are a helpful assistant."
        constraints = (
            "\nNote:\n"
            "1. Do not generate additional user input requests during task execution.\n"
            "2. Use only the available tools to perform actions.\n"
            "3. If a solution cannot be determined with the given tools or information, respond:\n"
            "\"I'm unable to complete this task with the current resources.\"\n"
            "4. Avoid infinite loops by stopping reasoning and returning an appropriate message if progress cannot be made.\n"
        )
        prompt_template = PromptTemplate.from_template(template, constraints=constraints)
        expected_template = template + constraints
        self.assertEqual(prompt_template.template, expected_template)

    def test_input_variables(self):
        template = "Answer the following question: {input}"
        constraints = "\nNote:\n1. Constraint 1.\n2. Constraint 2."
        prompt_template = PromptTemplate.from_template(template, constraints=constraints)
        self.assertIn("input", prompt_template.input_variables)

if __name__ == "__main__":
    unittest.main()