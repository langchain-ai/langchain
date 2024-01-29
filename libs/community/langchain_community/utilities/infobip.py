"""Util that calls Infobip."""
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


class InfobipAPIWrapper(BaseModel):
    """
    Wrapper around InfobipAPI.

    To use, you should have the ``infobip-api-python-sdk`` python package installed.
    https://pypi.org/project/infobip-api-python-sdk/
    This wrapper will use the Infobip API to send SMS or Email message.
    """

    sms_channel: Any  #: :meta private
    email_channel: Any  #: :meta private

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    from_email: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from infobip_channels.sms.channel import SMSChannel
            from infobip_channels.email.channel import EmailChannel
        except ImportError:
            raise ImportError(
                "Could not import infobip-api-python-sdk python package. "
                "Please install it with `pip install infobip-api-python-sdk`."
            )

        api_key = get_from_dict_or_env(
            values, "api_key", "INFOBIP_API_KEY"
        )
        values["api_key"] = api_key

        base_url = get_from_dict_or_env(
            values, "base_url", "INFOBIP_BASE_URL"
        )
        values["base_url"] = base_url

        auth_params: Dict = {
            "base_url": base_url,
            "api_key": api_key,
        }

        sms_channel: SmsChannel = SMSChannel.from_auth_params(auth_params)
        values["sms_channel"] = sms_channel

        email_channel: EmailChannel = EmailChannel.from_auth_params(auth_params)
        values["email_channel"] = email_channel
        return values

    def run(self, body: str, to: str, channel: str = "sms", subject: Optional[str] = "") -> str:
        if channel == "sms":
            if len(subject) > 0:
                body = f"{subject} - {body}"

            payload: Dict = {
                "messages": [
                    {
                        "destinations": [{"to": to}],
                        "text": body,
                    }
                ]
            }
            sms_response = self.sms_channel.send_sms_message(payload)
            return sms_response.messages[0].message_id
        elif channel == "email":
            payload: Dict = {
                "from": self.from_email,
                "to": to,
                "subject": subject,
                "text": body
            }
            email_response = self.email_channel.send_email_message(payload)
            return email_response.messages[0].message_id
        
        raise ValueError("Channel not supported, please use sms or email.")