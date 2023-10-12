"""Util that can interact with Zapier NLA.

Full docs here: https://nla.zapier.com/start/

Note: this wrapper currently only implemented the `api_key` auth method for testing
and server-side production use cases (using the developer's connected accounts on
Zapier.com)

For use-cases where LangChain + Zapier NLA is powering a user-facing application, and
LangChain needs access to the end-user's connected accounts on Zapier.com, you'll need
to use oauth. Review the full docs above and reach out to nla@zapier.com for
developer support.
"""
import json
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from requests import Request, Session

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env


class ZapierNLAWrapper(BaseModel):
    """Wrapper for Zapier NLA.

    Full docs here: https://nla.zapier.com/start/

    This wrapper supports both API Key and OAuth Credential auth methods. API Key
    is the fastest way to get started using this wrapper.

    Call this wrapper with either `zapier_nla_api_key` or
    `zapier_nla_oauth_access_token` arguments, or set the `ZAPIER_NLA_API_KEY`
    environment variable. If both arguments are set, the Access Token will take
    precedence.

    For use-cases where LangChain + Zapier NLA is powering a user-facing application,
    and LangChain needs access to the end-user's connected accounts on Zapier.com,
    you'll need to use OAuth. Review the full docs above to learn how to create
    your own provider and generate credentials.
    """

    zapier_nla_api_key: str
    zapier_nla_oauth_access_token: str
    zapier_nla_api_base: str = "https://nla.zapier.com/api/v1/"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _format_headers(self) -> Dict[str, str]:
        """Format headers for requests."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.zapier_nla_oauth_access_token:
            headers.update(
                {"Authorization": f"Bearer {self.zapier_nla_oauth_access_token}"}
            )
        else:
            headers.update({"X-API-Key": self.zapier_nla_api_key})

        return headers

    def _get_session(self) -> Session:
        session = requests.Session()
        session.headers.update(self._format_headers())
        return session

    async def _arequest(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Make an async request."""
        async with aiohttp.ClientSession(headers=self._format_headers()) as session:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

    def _create_action_payload(  # type: ignore[no-untyped-def]
        self, instructions: str, params: Optional[Dict] = None, preview_only=False
    ) -> Dict:
        """Create a payload for an action."""
        data = params if params else {}
        data.update(
            {
                "instructions": instructions,
            }
        )
        if preview_only:
            data.update({"preview_only": True})
        return data

    def _create_action_url(self, action_id: str) -> str:
        """Create a url for an action."""
        return self.zapier_nla_api_base + f"exposed/{action_id}/execute/"

    def _create_action_request(  # type: ignore[no-untyped-def]
        self,
        action_id: str,
        instructions: str,
        params: Optional[Dict] = None,
        preview_only=False,
    ) -> Request:
        data = self._create_action_payload(instructions, params, preview_only)
        return Request(
            "POST",
            self._create_action_url(action_id),
            json=data,
        )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""

        zapier_nla_api_key_default = None

        # If there is a oauth_access_key passed in the values
        # we don't need a nla_api_key it can be blank
        if "zapier_nla_oauth_access_token" in values:
            zapier_nla_api_key_default = ""
        else:
            values["zapier_nla_oauth_access_token"] = ""

        # we require at least one API Key
        zapier_nla_api_key = get_from_dict_or_env(
            values,
            "zapier_nla_api_key",
            "ZAPIER_NLA_API_KEY",
            zapier_nla_api_key_default,
        )

        values["zapier_nla_api_key"] = zapier_nla_api_key

        return values

    async def alist(self) -> List[Dict]:
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
        response = await self._arequest("GET", self.zapier_nla_api_base + "exposed/")
        return response["results"]

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
        https://nla.zapier.com/docs/using-the-api#ai-guessing)
        """
        session = self._get_session()
        try:
            response = session.get(self.zapier_nla_api_base + "exposed/")
            response.raise_for_status()
        except requests.HTTPError as http_err:
            if response.status_code == 401:
                if self.zapier_nla_oauth_access_token:
                    raise requests.HTTPError(
                        f"An unauthorized response occurred. Check that your "
                        f"access token is correct and doesn't need to be "
                        f"refreshed. Err: {http_err}",
                        response=response,
                    )
                raise requests.HTTPError(
                    f"An unauthorized response occurred. Check that your api "
                    f"key is correct. Err: {http_err}",
                    response=response,
                )
            raise http_err
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
        request = self._create_action_request(action_id, instructions, params)
        response = session.send(session.prepare_request(request))
        response.raise_for_status()
        return response.json()["result"]

    async def arun(
        self, action_id: str, instructions: str, params: Optional[Dict] = None
    ) -> Dict:
        """Executes an action that is identified by action_id, must be exposed
        (enabled) by the current user (associated with the set api_key). Change
        your exposed actions here: https://nla.zapier.com/demo/start/

        The return JSON is guaranteed to be less than ~500 words (350
        tokens) making it safe to inject into the prompt of another LLM
        call.
        """
        response = await self._arequest(
            "POST",
            self._create_action_url(action_id),
            json=self._create_action_payload(instructions, params),
        )
        return response["result"]

    def preview(
        self, action_id: str, instructions: str, params: Optional[Dict] = None
    ) -> Dict:
        """Same as run, but instead of actually executing the action, will
        instead return a preview of params that have been guessed by the AI in
        case you need to explicitly review before executing."""
        session = self._get_session()
        params = params if params else {}
        params.update({"preview_only": True})
        request = self._create_action_request(action_id, instructions, params, True)
        response = session.send(session.prepare_request(request))
        response.raise_for_status()
        return response.json()["input_params"]

    async def apreview(
        self, action_id: str, instructions: str, params: Optional[Dict] = None
    ) -> Dict:
        """Same as run, but instead of actually executing the action, will
        instead return a preview of params that have been guessed by the AI in
        case you need to explicitly review before executing."""
        response = await self._arequest(
            "POST",
            self._create_action_url(action_id),
            json=self._create_action_payload(instructions, params, preview_only=True),
        )
        return response["result"]

    def run_as_str(self, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """Same as run, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        data = self.run(*args, **kwargs)
        return json.dumps(data)

    async def arun_as_str(self, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """Same as run, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        data = await self.arun(*args, **kwargs)
        return json.dumps(data)

    def preview_as_str(self, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """Same as preview, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        data = self.preview(*args, **kwargs)
        return json.dumps(data)

    async def apreview_as_str(  # type: ignore[no-untyped-def]
        self, *args, **kwargs
    ) -> str:
        """Same as preview, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        data = await self.apreview(*args, **kwargs)
        return json.dumps(data)

    def list_as_str(self) -> str:  # type: ignore[no-untyped-def]
        """Same as list, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        actions = self.list()
        return json.dumps(actions)

    async def alist_as_str(self) -> str:  # type: ignore[no-untyped-def]
        """Same as list, but returns a stringified version of the JSON for
        insertting back into an LLM."""
        actions = await self.alist()
        return json.dumps(actions)
