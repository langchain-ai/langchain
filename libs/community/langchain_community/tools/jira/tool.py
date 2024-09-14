"""
This tool allows agents to interact with the atlassian-python-api library
and operate on a Jira instance. For more information on the
atlassian-python-api library, see https://atlassian-python-api.readthedocs.io/jira.html

To use this tool, you must first set as environment variables:
    JIRA_API_TOKEN
    JIRA_USERNAME
    JIRA_INSTANCE_URL
    JIRA_CLOUD

Below is a sample script that uses the Jira tool:

```python
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper

jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
```
"""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.utilities.jira import JiraAPIWrapper


class JiraAction(BaseTool):
    """Tool that queries the Atlassian Jira API."""

    api_wrapper: JiraAPIWrapper = Field(default_factory=JiraAPIWrapper)  # type: ignore[arg-type]
    mode: str
    name: str = ""
    description: str = ""

    def _run(
        self,
        instructions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Atlassian Jira API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)
