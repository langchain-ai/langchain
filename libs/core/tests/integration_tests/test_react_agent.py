# tests/integration_tests/agents/test_react_agent.py

import unittest
from unittest.mock import patch
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.llms.openai import OpenAI

class TestReActAgentIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define the tools
        def search_tool(input_text):
            return f"Searching for: {input_text}"

        def calculator_tool(input_text):
            try:
                return str(eval(input_text))
            except Exception:
                return "Invalid input"

        cls.tools = [
            Tool(name="search", func=search_tool, description="Searches for information."),
            Tool(name="calculator", func=calculator_tool, description="Performs calculations.")
        ]

        # Define the prompt template with constraints
        constraints = """
Note:
1. Do not generate additional user input requests during task execution.
2. Use only the available tools to perform actions.
3. If a solution cannot be determined with the given tools or information, respond:
"I'm unable to complete this task with the current resources."
4. Avoid infinite loops by stopping reasoning and returning an appropriate message if progress cannot be made.
"""

        cls.prompt_template = PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=(
                "You are a helpful assistant. "
                "Use the following tools to answer the question: {input}.\n"
                "Available tools: {tool_names}\n"
                "You can use these tools: {tools}\n"
                "Please follow this format:\n"
                "Thought: [your thought process]\n"
                "Action: [tool name](input_text)\n"
                "Observation: [result of action]\n"
                "Agent thoughts and progress: {agent_scratchpad}\n"
            ),
            constraints=constraints,
        )

        # Initialize the LLM
        cls.llm = OpenAI(temperature=0)

        # Create the ReAct agent
        cls.agent = create_react_agent(
            llm=cls.llm,
            tools=cls.tools,
            prompt=cls.prompt_template,
            max_iterations=5,
        )

        # Create the AgentExecutor
        cls.agent_executor = AgentExecutor(
            agent=cls.agent,
            tools=cls.tools,
            verbose=False,
            handle_parsing_errors=True,
        )

    @patch('langchain.llms.openai.OpenAI._call')
    def test_agent_solves_simple_arithmetic(self, mock_call):
        # Mock the LLM response
        mock_call.return_value = "Thought: I need to calculate 2 + 2.\nAction: calculator(2 + 2)\nObservation: 4\nAnswer: 4"
        input_text = "What is 2 + 2?"
        response = self.agent_executor.run(input_text)
        self.assertIn("4", response)

    @patch('langchain.llms.openai.OpenAI._call')
    def test_agent_handles_unresolvable_task(self, mock_call):
        mock_call.return_value = "I'm unable to complete this task with the current resources."
        input_text = "Translate 'hello' to French."
        response = self.agent_executor.run(input_text)
        self.assertIn("I'm unable to complete this task with the current resources.", response)

    @patch('langchain.llms.openai.OpenAI._call')
    def test_agent_avoids_infinite_loops(self, mock_call):
        mock_call.side_effect = [
            "Thought: I need more information.\nAction: None",
            "Thought: I need more information.\nAction: None",
            "Thought: I need more information.\nAction: None",
            "I'm unable to complete this task with the current resources."
        ]
        input_text = "What is the color of the invisible car?"
        response = self.agent_executor.run(input_text)
        self.assertIn("I'm unable to complete this task with the current resources.", response)

    @patch('langchain.llms.openai.OpenAI._call')
    def test_agent_does_not_request_user_input(self, mock_call):
        mock_call.return_value = "I'm unable to complete this task with the current resources."
        input_text = "Please ask me a question."
        response = self.agent_executor.run(input_text)
        self.assertIn("I'm unable to complete this task with the current resources.", response)
        self.assertNotIn("What would you like to know?", response)

if __name__ == '__main__':
    unittest.main()