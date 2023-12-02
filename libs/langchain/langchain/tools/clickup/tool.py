"""
This tool allows agents to interact with the clickup library
and operate on a Clickup instance.
To use this tool, you must first set as environment variables:
    client_secret
    client_id
    code

Below is a sample script that uses the Clickup tool:

```python
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.clickup.toolkit import ClickupToolkit
from langchain.llms import OpenAI
from langchain.utilities.clickup import ClickupAPIWrapper

llm = OpenAI(temperature=0)
clickup = ClickupAPIWrapper()
toolkit = ClickupToolkit.from_clickup_api_wrapper(clickup)
agent = initialize_agent(
    toolkit.get_tools(),
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```
"""
from typing import Optional

from langchain_core.pydantic_v1 import Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utilities.clickup import ClickupAPIWrapper


class ClickupAction(BaseTool):
    """Tool that queries the  Clickup API."""

    api_wrapper: ClickupAPIWrapper = Field(default_factory=ClickupAPIWrapper)
    mode: str
    name: str = ""
    description: str = ""

    def _run(
        self,
        instructions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the  Clickup API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)
