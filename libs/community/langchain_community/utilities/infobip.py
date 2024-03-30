"""Util that sends messages via Infobip."""
from typing import Dict, List, Optional

import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class InfobipAPIWrapper(BaseModel):
    """Wrapper for Infobip API for messaging."""

    infobip_api_key: Optional[str] = None
    infobip_base_url: Optional[str] = "https://api.infobip.com"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["infobip_api_key"] = get_from_dict_or_env(
            values, "infobip_api_key", "INFOBIP_API_KEY"
        )
        values["infobip_base_url"] = get_from_dict_or_env(
            values, "infobip_base_url", "INFOBIP_BASE_URL"
        )
        return values

    def _get_requests_session(self) -> requests.Session:
        """Get a requests session with the correct headers."""
        retry_strategy: Retry = Retry(
            total=4,  # Maximum number of retries
            backoff_factor=2,  # Exponential backoff factor
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        )
        adapter: HTTPAdapter = HTTPAdapter(max_retries=retry_strategy)

        session = requests.Session()
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Authorization": f"App {self.infobip_api_key}",
                "User-Agent": "infobip-langchain-community",
            }
        )
        return session

    def _send_sms(
        self, sender: str, destination_phone_numbers: List[str], text: str
    ) -> str:
        """Send an SMS message."""
        json: Dict = {
            "messages": [
                {
                    "destinations": [
                        {"to": destination} for destination in destination_phone_numbers
                    ],
                    "from": sender,
                    "text": text,
                }
            ]
        }

        session: requests.Session = self._get_requests_session()
        session.headers.update(
            {
                "Content-Type": "application/json",
            }
        )

        response: requests.Response = session.post(
            f"{self.infobip_base_url}/sms/2/text/advanced",
            json=json,
        )

        response_json: Dict = response.json()
        try:
            if response.status_code != 200:
                return response_json["requestError"]["serviceException"]["text"]
        except KeyError:
            return "Failed to send message"

        try:
            return response_json["messages"][0]["messageId"]
        except KeyError:
            return (
                "Could not get message ID from response, message was sent successfully"
            )

    def _send_email(
        self, from_email: str, to_email: str, subject: str, body: str
    ) -> str:
        """Send an email message."""

        try:
            from requests_toolbelt import MultipartEncoder
        except ImportError as e:
            raise ImportError(
                "Unable to import requests_toolbelt, please install it with "
                "`pip install -U requests-toolbelt`."
            ) from e
        form_data: Dict = {
            "from": from_email,
            "to": to_email,
            "subject": subject,
            "text": body,
        }

        data = MultipartEncoder(fields=form_data)

        session: requests.Session = self._get_requests_session()
        session.headers.update(
            {
                "Content-Type": data.content_type,
            }
        )

        response: requests.Response = session.post(
            f"{self.infobip_base_url}/email/3/send",
            data=data,
        )

        response_json: Dict = response.json()

        try:
            if response.status_code != 200:
                return response_json["requestError"]["serviceException"]["text"]
        except KeyError:
            return "Failed to send message"

        try:
            return response_json["messages"][0]["messageId"]
        except KeyError:
            return (
                "Could not get message ID from response, message was sent successfully"
            )

    def run(
        self,
        body: str = "",
        to: str = "",
        sender: str = "",
        subject: str = "",
        channel: str = "sms",
    ) -> str:
        if channel == "sms":
            if sender == "":
                raise ValueError("Sender must be specified for SMS messages")

            if to == "":
                raise ValueError("Destination must be specified for SMS messages")

            if body == "":
                raise ValueError("Body must be specified for SMS messages")

            return self._send_sms(
                sender=sender,
                destination_phone_numbers=[to],
                text=body,
            )
        elif channel == "email":
            if sender == "":
                raise ValueError("Sender must be specified for email messages")

            if to == "":
                raise ValueError("Destination must be specified for email messages")

            if subject == "":
                raise ValueError("Subject must be specified for email messages")

            if body == "":
                raise ValueError("Body must be specified for email messages")

            return self._send_email(
                from_email=sender,
                to_email=to,
                subject=subject,
                body=body,
            )
        else:
            raise ValueError(f"Channel {channel} is not supported")
