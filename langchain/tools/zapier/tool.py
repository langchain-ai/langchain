"""## Zapier Natural Language Actions API
\
Full docs here: https://nla.zapier.com/api/v1/docs

**Zapier Natural Language Actions** gives you access to the 5k+ apps, 20k+ actions
on Zapier's platform through a natural language API interface.

NLA supports apps like Gmail, Salesforce, Trello, Slack, Asana, HubSpot, Google Sheets,
Microsoft Teams, and thousands more apps: https://zapier.com/apps

Zapier NLA handles ALL the underlying API auth and translation from
natural language --> underlying API call --> return simplified output for LLMs
The key idea is you, or your users, expose a set of actions via an oauth-like setup
window, which you can then query and execute via a REST API.

NLA offers both API Key and OAuth for signing NLA API requests.

1. Server-side (API Key): for quickly getting started, testing, and production scenarios
    where LangChain will only use actions exposed in the developer's Zapier account
    (and will use the developer's connected accounts on Zapier.com)

2. User-facing (Oauth): for production scenarios where you are deploying an end-user
    facing application and LangChain needs access to end-user's exposed actions and
    connected accounts on Zapier.com

This quick start will focus on the server-side use case for brevity.
Review [full docs](https://nla.zapier.com/api/v1/docs) or reach out to
nla@zapier.com for user-facing oauth developer support.

Typically you'd use SequentialChain, here's a basic example:

    1. Use NLA to find an email in Gmail
    2. Use LLMChain to generate a draft reply to (1)
    3. Use NLA to send the draft reply (2) to someone in Slack via direct message

In code, below:

```python

import os

# get from https://platform.openai.com/
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

# get from https://nla.zapier.com/demo/provider/debug
# (under User Information, after logging in):
os.environ["ZAPIER_NLA_API_KEY"] = os.environ.get("ZAPIER_NLA_API_KEY", "")

from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper

## step 0. expose gmail 'find email' and slack 'send channel message' actions

# first go here, log in, expose (enable) the two actions:
#    https://nla.zapier.com/demo/start
#    -- for this example, can leave all fields "Have AI guess"
# in an oauth scenario, you'd get your own <provider> id (instead of 'demo')
# which you route your users through first

llm = OpenAI(temperature=0)
zapier = ZapierNLAWrapper()
## To leverage a nla_oauth_access_token you may pass the value to the ZapierNLAWrapper
## If you do this there is no need to initialize the ZAPIER_NLA_API_KEY env variable
# zapier = ZapierNLAWrapper(zapier_nla_oauth_access_token="TOKEN_HERE")
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(
    toolkit.get_tools(),
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run(("Summarize the last email I received regarding Silicon Valley Bank. "
    "Send the summary to the #test-zapier channel in slack."))
```

"""
from typing import Any, Dict, Optional

from pydantic import Field, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.tools.zapier.prompt import BASE_ZAPIER_TOOL_PROMPT
from langchain.utilities.zapier import ZapierNLAWrapper


class ZapierNLARunAction(BaseTool):
    """
    Args:
        action_id: a specific action ID (from list actions) of the action to execute
            (the set api_key must be associated with the action owner)
        instructions: a natural language instruction string for using the action
            (eg. "get the latest email from Mike Knoop" for "Gmail: find email" action)
        params: a dict, optional. Any params provided will *override* AI guesses
            from `instructions` (see "understanding the AI guessing flow" here:
            https://nla.zapier.com/api/v1/docs)
    """

    api_wrapper: ZapierNLAWrapper = Field(default_factory=ZapierNLAWrapper)
    action_id: str
    params: Optional[dict] = None
    zapier_description: str
    params_schema: Dict[str, str] = Field(default_factory=dict)
    name = ""
    description = ""

    @root_validator
    def set_name_description(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        zapier_description = values["zapier_description"]
        params_schema = values["params_schema"]
        if "instructions" in params_schema:
            del params_schema["instructions"]
        values["name"] = zapier_description
        values["description"] = BASE_ZAPIER_TOOL_PROMPT.format(
            zapier_description=zapier_description,
            params=str(list(params_schema.keys())),
        )
        return values

    def _run(
        self, instructions: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        return self.api_wrapper.run_as_str(self.action_id, instructions, self.params)

    async def _arun(
        self,
        _: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        raise NotImplementedError("ZapierNLAListActions does not support async")


ZapierNLARunAction.__doc__ = (
    ZapierNLAWrapper.run.__doc__ + ZapierNLARunAction.__doc__  # type: ignore
)


# other useful actions


class ZapierNLAListActions(BaseTool):
    """
    Args:
        None
    """

    name = "Zapier NLA: List Actions"
    description = BASE_ZAPIER_TOOL_PROMPT + (
        "This tool returns a list of the user's exposed actions."
    )
    api_wrapper: ZapierNLAWrapper = Field(default_factory=ZapierNLAWrapper)

    def _run(
        self,
        _: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        return self.api_wrapper.list_as_str()

    async def _arun(
        self,
        _: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        raise NotImplementedError("ZapierNLAListActions does not support async")


ZapierNLAListActions.__doc__ = (
    ZapierNLAWrapper.list.__doc__ + ZapierNLAListActions.__doc__  # type: ignore
)
