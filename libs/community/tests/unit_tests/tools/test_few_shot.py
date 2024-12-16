from typing import Type

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_community.tools.few_shot.tool import FewShotSQLTool


class TestFewShotSQLToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[FewShotSQLTool]:
        return FewShotSQLTool

    @property
    def tool_constructor_params(self) -> dict:
        examples = [
            {
                "input": "Number of rows in artist table",
                "output": "select count(*) from Artist",
            },
            {
                "input": "Number of rows in album table",
                "output": "select count(*) from Album",
            },
        ]
        embeddings = DeterministicFakeEmbedding(size=10)

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embeddings,
            InMemoryVectorStore,
            k=5,
            input_keys=["input"],
        )
        return {
            "example_selector": example_selector,
            "description": "Use this tool to select examples.",
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"question": "How many rows are in the customer table?"}
