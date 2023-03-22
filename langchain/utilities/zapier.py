"""Util that can interact with Zapier NLA.

Full docs here: https://nla.zapier.com/api/v1/docs

Note: this wrapper currently only implemented the `api_key` auth method for testing
and server-side production use cases (using the developer's connected accounts on
Zapier.com)

For use-cases where LangChain + Zapier NLA is powering a user-facing application, and
LangChain needs access to the end-user's connected accounts on Zapier.com, you'll need
to use oauth. Review the full docs above and reach out to nla@zapier.com for
developer support.
"""
import json
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Extra, root_validator
from requests import Request, Session

from langchain.utils import get_from_dict_or_env


class ZapierNLAWrapper(BaseModel):
    """Wrapper for Zapier NLA.

    Full docs here: https://nla.zapier.com/api/v1/docs

    Note: this wrapper currently only implemented the `api_key` auth method for
    testingand server-side production use cases (using the developer's connected
    accounts on Zapier.com)

    For use-cases where LangChain + Zapier NLA is powering a user-facing application,
    and LangChain needs access to the end-user's connected accounts on Zapier.com,
    you'll need to use oauth. Review the full docs above and reach out to
    nla@zapier.com for developer support.
    """

    zapier_nla_api_key: str
    zapier_nla_api_base: str = "https://nla.zapier.com/api/v1/"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _get_session(self) -> Session:
        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        session.params = {"api_key": self.zapier_nla_api_key}
        return session

    def _get_action_request(
        self, action_id: str, instructions: str, params: Optional[Dict] = None
    ) -> Request:
        data = params if params else {}
        data.update(
            {
                "instructions": instructions,
            }
        )
        return Request(
            "POST",
            self.zapier_nla_api_base + f"exposed/{action_id}/execute/",
            json=data,
        )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        zapier_nla_api_key = get_from_dict_or_env(
            values, "zapier_nla_api_key", "ZAPIER_NLA_API_KEY"
        )
        values["zapier_nla_api_key"] = zapier_nla_api_key

        return values

    def list(self) -> List[Dict]:
        """Returns a list of all exposed (enabled) actions associated with
        current user (associated with the set api_key). Change your exposed
        actions here: https://nla.zapier.com/demo/start/

        The return list can be empty if no actions exposed. Else will contain
        a list of action objects:

        [{
            "id": str,
            "description": str,
            "params": Dict[str, str]
        }]

        `params` will always contain an `instructions` key, the only required
        param. All others optional and if provided will override any AI guesses
        (see "understanding the AI guessing flow" here:
        https://nla.zapier.com/api/v1/docs)
        """
        session = self._get_session()
        response = session.get(self.zapier_nla_api_base + "exposed/")
        response.raise_for_status()
        return response.json()["results"]

    def run(
        self, action_id: str, instructions: str, params: Optional[Dict] = None
    ) -> Dict:
        """Executes an action that is identified by action_id, must be exposed
        (enabled) by the current user (associated with the set api_key). Change
        your exposed actions here: https://nla.zapier.com/demo/start/

        The return JSON is guaranteed to be less than ~500 words (350
        tokens) making it safe to inject into the prompt of another LLM
        call.
        """
        session = self._get_session()
        request = self._get_action_request(action_id, instructions, params)
        response = session.send(session.prepare_request(request))
        response.raise_for_status()
        return response.json()["result"]

    def preview(
        self, action_id: str, instructions: str, params: Optional[Dict] = None
    ) -> Dict:
        """Same as run, but instead of actually executing the action, will
        instead return a preview of params that have been guessed by the AI in
        case you need to explicitly review before executing."""
        session = self._get_session()
        params = params if params else {}
        params.update({"preview_only": True})
        request = self._get_action_request(action_id, instructions, params)
        response = session.send(session.prepare_request(request))
        response.raise_for_status()
        return response.json()["input_params"]

    def run_as_str(self, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """Same as run, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        data = self.run(*args, **kwargs)
        return json.dumps(data)

    def preview_as_str(self, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """Same as preview, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        data = self.preview(*args, **kwargs)
        return json.dumps(data)

    def list_as_str(self) -> str:  # type: ignore[no-untyped-def]
        """Same as list, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        actions = self.list()
        return json.dumps(actions)
