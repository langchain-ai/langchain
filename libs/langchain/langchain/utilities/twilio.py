"""Util that calls Twilio."""
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


class TwilioAPIWrapper(BaseModel):
    """Messaging Client using Twilio.

    To use, you should have the ``twilio`` python package installed,
    and the environment variables ``TWILIO_ACCOUNT_SID``, ``TWILIO_AUTH_TOKEN``, and
    ``TWILIO_FROM_NUMBER``, or pass `account_sid`, `auth_token`, and `from_number` as
    named parameters to the constructor.

    Example:
        .. code-block:: python

            from langchain.utilities.twilio import TwilioAPIWrapper
            twilio = TwilioAPIWrapper(
                account_sid="ACxxx",
                auth_token="xxx",
                from_number="+10123456789"
            )
            twilio.run('test', '+12484345508')
    """

    client: Any  #: :meta private:
    account_sid: Optional[str] = None
    """Twilio account string identifier."""
    auth_token: Optional[str] = None
    """Twilio auth token."""
    from_number: Optional[str] = None
    """A Twilio phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) 
        format, an 
        [alphanumeric sender ID](https://www.twilio.com/docs/sms/send-messages#use-an-alphanumeric-sender-id), 
        or a [Channel Endpoint address](https://www.twilio.com/docs/sms/channels#channel-addresses) 
        that is enabled for the type of message you want to send. Phone numbers or 
        [short codes](https://www.twilio.com/docs/sms/api/short-code) purchased from 
        Twilio also work here. You cannot, for example, spoof messages from a private 
        cell phone number. If you are using `messaging_service_sid`, this parameter 
        must be empty.
    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from twilio.rest import Client
        except ImportError:
            raise ImportError(
                "Could not import twilio python package. "
                "Please install it with `pip install twilio`."
            )
        account_sid = get_from_dict_or_env(values, "account_sid", "TWILIO_ACCOUNT_SID")
        auth_token = get_from_dict_or_env(values, "auth_token", "TWILIO_AUTH_TOKEN")
        values["from_number"] = get_from_dict_or_env(
            values, "from_number", "TWILIO_FROM_NUMBER"
        )
        values["client"] = Client(account_sid, auth_token)
        return values

    def run(self, body: str, to: str) -> str:
        """Run body through Twilio and respond with message sid.

        Args:
            body: The text of the message you want to send. Can be up to 1,600
                characters in length.
            to: The destination phone number in
                [E.164](https://www.twilio.com/docs/glossary/what-e164) format for
                SMS/MMS or
                [Channel user address](https://www.twilio.com/docs/sms/channels#channel-addresses)
                for other 3rd-party channels.
        """  # noqa: E501
        message = self.client.messages.create(to, from_=self.from_number, body=body)
        return message.sid
