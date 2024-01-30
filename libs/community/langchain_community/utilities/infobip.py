"""Util that sends messages via Infobip."""
from typing import Dict, List, Optional

import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.adapters import HTTPAdapter
from requests_toolbelt import MultipartEncoder
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
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Authorization": f"App {self.infobip_api_key}",
                "User-Agent": "langchain-community",
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
        response.raise_for_status()

        return response.json()["messages"][0]["messageId"]

    def _send_email(
        self, from_email: str, to_email: str, subject: str, body: str
    ) -> str:
        """Send an email message."""
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

        response.raise_for_status()
        return response.json()["messages"][0]["messageId"]

    def _validate_email_address(self, email_address: str) -> str:
        """Validate an email address."""
        json: Dict = {
            "to": email_address,
        }

        session: requests.Session = self._get_requests_session()
        session.headers.update(
            {
                "Content-Type": "application/json",
            }
        )

        response: requests.Response = session.post(
            f"{self.infobip_base_url}/email/2/validation",
            json=json,
        )

        response.raise_for_status()
        validation_status: bool = response.json()["validMailbox"]
        return str(validation_status)

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
        elif channel == "email-validation":
            if to == "":
                raise ValueError("Destination must be specified for email validation")

            return self._validate_email_address(
                email_address=to,
            )
        else:
            raise ValueError(f"Channel {channel} is not supported")
