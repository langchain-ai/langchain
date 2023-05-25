"""Util that calls Twilio."""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


class Sms(BaseModel):
    """Sms Client using Twilio.

    To use, you should have the ``twilio`` python package installed,
    and the environment variables ``ACCOUNT_SID``, ``AUTH_TOKEN``, and ``FROM_NUMBER``, or pass
    `account_sid`, `auth_token`, and `from_number` as named parameters to the constructor.

    Example:
        .. code-block:: python

            from langchain.utilities.sms import Sms
            client = Sms(account_sid="ACxxx", auth_token="xxx", from_number="+15551234567")
            client.run('test', '+15551234567')
    """

    sms_client: Any  #: :meta private:
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None
    from_number: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["account_sid"] = get_from_dict_or_env(
            values, "account_sid", "ACCOUNT_SID"
        )
        values["auth_token"] = get_from_dict_or_env(values, "auth_token", "AUTH_TOKEN")
        values["from_number"] = get_from_dict_or_env(
            values, "from_number", "FROM_NUMBER"
        )
        try:
            from twilio.rest import Client

            values["sms_client"] = Client
        except ImportError:
            raise ValueError(
                "Could not import twilio python package. "
                "Please install it with `pip3 install twilio`."
            )
        return values

    def run(self, body: str, to: str, **kwargs: Any) -> str:
        """Run body through Twilio and respond with message sid."""
        return self.sms(body, to)

    async def arun(self, body: str, to: str, **kwargs: Any) -> str:
        """Run message through Twilio and respond with message sid async."""
        raise NotImplementedError("Sms does not yet support async")

    def sms(self, body: str, to: str) -> str:
        """Send message through Twilio and return the message sid."""
        client = self.sms_client(self.account_sid, self.auth_token)
        message = client.messages.create(to=to, from_=self.from_number, body=body)
        return message.sid
