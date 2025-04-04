from typing import Type

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_community.tools.few_shot.tool import FewShotSQLTool

EXAMPLES = [
    {
        "input": "Number of rows in artist table",
        "output": "select count(*) from Artist",
    },
    {
        "input": "Number of rows in album table",
        "output": "select count(*) from Album",
    },
]
EMBEDDINGS = DeterministicFakeEmbedding(size=10)

EXAMPLE_SELECTOR = SemanticSimilarityExampleSelector.from_examples(
    EXAMPLES,
    EMBEDDINGS,
    InMemoryVectorStore,
    k=5,
    input_keys=["input"],
)


class TestFewShotSQLToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[FewShotSQLTool]:
        return FewShotSQLTool

    @property
    def tool_constructor_params(self) -> dict:
        return {
            "example_selector": EXAMPLE_SELECTOR,
            "description": "Use this tool to select examples.",
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"question": "How many rows are in the customer table?"}


class TestFewShotSQLToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[FewShotSQLTool]:
        return FewShotSQLTool

    @property
    def tool_constructor_params(self) -> dict:
        return {
            "example_selector": EXAMPLE_SELECTOR,
            "description": "Use this tool to select examples.",
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"question": "How many rows are in the customer table?"}
