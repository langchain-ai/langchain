from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class _FewShotToolInput(BaseModel):
    question: str = Field(
        ..., description="The question for which we want example SQL queries."
    )


class FewShotSQLTool(BaseTool):
    """Tool to get example SQL queries related to an input question."""

    name: str = "few_shot_sql"
    description: str = "Tool to get example SQL queries related to an input question."
    args_schema: Type[BaseModel] = _FewShotToolInput

    example_selector: BaseExampleSelector = Field(exclude=True)
    example_input_key: str = "input"
    example_query_key: str = "query"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        example_prompt = PromptTemplate.from_template(
            f"User input: {self.example_input_key}\nSQL query: {self.example_query_key}"
        )
        prompt = FewShotPromptTemplate(
            example_prompt=example_prompt,
            example_selector=self.example_selector,
            suffix="",
            input_variables=[self.example_input_key],
        )
        return prompt.format(**{self.example_input_key: question})
