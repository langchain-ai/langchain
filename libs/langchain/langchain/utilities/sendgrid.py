"""Util that calls Sendgrid."""
from typing import Any, Dict, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env


class SendgridAPIWrapper(BaseModel):
    """Email Client using Sendgrid.

    To use, you should have the ``sendgrid`` python package installed,
    and the environment variable ``SENDGRID_API_KEY`` , or pass api_key
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.utilities.sendgrid import SendgridAPIWrapper
            sendgrid = SendgridAPIWrapper(
                api_key="SG.xxx"
            )
            sendgrid.run("from@email.com","to@email.com","subject","content","text/plain")
    """

    client: Any  #: :meta private:
    api_key: Optional[str] = None
    """Sendgrid API key."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import sendgrid
        except ImportError:
            raise ImportError(
                "Could not import sendgrid python package. "
                "Please install it with `pip install sendgrid`."
            )
        api_key = get_from_dict_or_env(values, "api_key", "SENDGRID_API_KEY")

        values["client"] = sendgrid.SendGridAPIClient(api_key=api_key)
        return values

    def run(
        self,
        from_email: str,
        to_email: str,
        subject: str,
        content: str,
        content_type: str,
    ) -> str:
        """Run body through Sendgrid and respond with status code.

        Args:
            from_email: The sender of the email.
            to_email: The destination of the email.
            subject: The subject of the email.
            content: The content of the email.
            content_type: The type of content of the email.
        """
        try:
            from sendgrid.helpers.mail import Content, Email, Mail, To
        except ImportError:
            raise ImportError(
                "Could not import sendgrid python package. "
                "Please install it with `pip install sendgrid`."
            )
        from_email = Email(from_email)
        to_email = To(to_email)
        content = Content(content_type, content)
        mail = Mail(from_email, to_email, subject, content)
        response = self.client.send(mail)

        return response.status_code
