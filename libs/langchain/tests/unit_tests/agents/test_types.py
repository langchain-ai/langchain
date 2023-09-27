import unittest

from langchain.agents.agent_types import AgentType, AGENT_TO_CLASS


class TestTypes(unittest.TestCase):
    def test_confirm_full_coverage(self) -> None:
        self.assertEqual(list(AgentType), list(AGENT_TO_CLASS.keys()))
