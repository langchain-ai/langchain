"""
This tool allows agents to interact with the pygithub library
and operate on a GitHub repository.

To use this tool, you must first set as environment variables:
    GITHUB_API_TOKEN
    GITHUB_REPOSITORY -> format: {owner}/{repo}

TODO: remove below
Below is a sample script that uses the Github tool:

```python
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.github.toolkit import GitHubToolkit
from langchain.llms import OpenAI
from langchain.utilities.github import GitHubAPIWrapper

llm = OpenAI(temperature=0)
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "{{Enter a prompt here to direct the agent}}"
)

```
"""
from typing import Optional

from pydantic import Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.github import GitHubAPIWrapper


class GitHubAction(BaseTool):
    api_wrapper: GitHubAPIWrapper = Field(default_factory=GitHubAPIWrapper)
    mode: str
    name = ""
    description = ""

    def _run(
        self,
        instructions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the GitHub API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)

    async def _arun(
        self,
        _: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the GitHub API to run an operation."""
        raise NotImplementedError("GitHubAction does not support async")
