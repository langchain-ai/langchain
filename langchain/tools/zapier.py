"""## Zapier Natural Language Actions API
\
Full docs here: https://nla.zapier.com/api/v1/dynamic/docs

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
Review [full docs](https://nla.zapier.com/api/v1/dynamic/docs) or reach out to
nla@zapier.com for user-facing oauth developer support.

Typically you'd use SequentialChain, here's a basic example:

    1. Use NLA to find an email in Gmail
    2. Use LLMChain to generate a draft reply to (1)
    3. Use NLA to send the draft reply (2) to someone in Slack via direct mesage
    
In code, below:

```python

import os

# get from https://platform.openai.com/
os.environ["OPENAI_API_KEY"] = "..."

# get from https://nla.zapier.com/demo/provider/debug (under User, after logging in):
os.environ["ZAPIER_NLA_API_KEY"] = "..."

from langchain.llms import OpenAI
from langchain.chains import LLMChain, TransformChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.tools.zapier import ZapierNLAListActions, ZapierNLARunAction
from langchain.utilities.zapier import ZapierNLAWrapper

## step 0. expose gmail 'find email' and slack 'send channel message' actions

# first go here, log in, expose (enable) the two actions:
#    https://nla.zapier.com/demo/start
#    -- for this example, can leave all fields "Have AI guess"
# in an oauth scenario, you'd get your own <provider> id (instead of 'demo')
# which you route your users through first

actions = ZapierNLAWrapper().list()

## step 1. gmail find email

GMAIL_SEARCH_INSTRUCTIONS = "Grab the latest email from Bryan Helmig"

def nla_gmail(inputs):
    action = next((
        a for a in actions if a["description"].startswith("Gmail: Find Email")
    ), None)
    data = ZapierNLARunAction(action_id=action["id"]).run(inputs["instructions"])
    return {
        "email_data": data
    }
gmail_chain = TransformChain(
    input_variables=["instructions"],
    output_variables=["email_data"],
    transform=nla_gmail
)

## step 2. generate draft reply

template = \"""You are an assisstant who drafts replies to an incoming email.
Output draft reply in plain text (not JSON).

Incoming email:
{email_data}

Draft email reply:\"""

prompt_template = PromptTemplate(input_variables=["email_data"], template=template)
reply_chain = LLMChain(llm=OpenAI(temperature=.7), prompt=prompt_template)

## step 3. send draft reply via a slack direct message

SLACK_HANDLE = "@knoop"

def nla_slack(inputs):
    action = next(
        (a for a in actions if a["description"].startswith("Slack: Send Direct Message")
    ), None)
    instructions = f'Send this to {SLACK_HANDLE} in Slack: {inputs["draft_reply"]}'
    return {"slack_data": ZapierNLARunAction(action_id=action["id"]).run(instructions)}
slack_chain = TransformChain(
    input_variables=["draft_reply"],
    output_variables=["slack_data"],
    transform=nla_slack
)

## finally, execute

overall_chain = SimpleSequentialChain(
    chains=[gmail_chain, reply_chain, slack_chain],
    verbose=True
)
overall_chain.run(GMAIL_SEARCH_INSTRUCTIONS)


```

"""
from typing import Any, Dict, Optional

from pydantic import Field, root_validator

from langchain.tools.base import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper

zapier_nla_base_desc = (
    "A wrapper around Zapier NLA actions. "
    "The input to this tool is a natural language instruction, "
    'for example "get the latest email from my bank" or '
    '"send a slack message to the #general channel". '
    "This tool specifically used for: "
)


class ZapierNLARunAction(BaseTool):
    """
    Args:
        action_id: a specific action ID (from list actions) of the action to execute
            (the set api_key must be associated with the action owner)
        instructions: a natural language instruction string for using the action
            (eg. "get the latest email from Mike Knoop" for "Gmail: find email" action)
        params: a dict, optional. Any params provided will *override* AI guesses
            from `instructions` (see "understanding the AI guessing flow" here:
            https://nla.zapier.com/api/v1/dynamic/docs)
    """

    api_wrapper: ZapierNLAWrapper = Field(default_factory=ZapierNLAWrapper)
    action_id: str
    params: Optional[dict] = None
    zapier_description: str
    name = ""
    description = ""

    @root_validator
    def set_name_description(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        zapier_description = values["zapier_description"]
        values["name"] = zapier_description
        values["description"] = zapier_nla_base_desc + zapier_description
        return values

    def _run(self, instructions: str) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        return self.api_wrapper.run_as_str(self.action_id, instructions, self.params)

    async def _arun(self, _: str) -> str:
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
    description = zapier_nla_base_desc + (
        "This tool returns a list of the user's exposed actions."
    )
    api_wrapper: ZapierNLAWrapper = Field(default_factory=ZapierNLAWrapper)

    def _run(self, _: str) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        return self.api_wrapper.list_as_str()

    async def _arun(self, _: str) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        raise NotImplementedError("ZapierNLAListActions does not support async")


ZapierNLAListActions.__doc__ = (
    ZapierNLAWrapper.list.__doc__ + ZapierNLAListActions.__doc__  # type: ignore
)
