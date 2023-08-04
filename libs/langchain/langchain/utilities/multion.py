"""Util that calls MultiOn Client.

In order to set this up, follow instructions at:
https://multion.notion.site/Download-MultiOn-ddddcfe719f94ab182107ca2612c07a5
"""
from typing import Any, Optional

from pydantic import BaseModel


class MultionAPI:
    def __init__(self) -> None:
        self.tabId = None
        self.new_session_count = 0

    def create_session(self, query: str, url: str) -> str:
        """Always the first step to run any activities that can be done using browser.

        Args:
            'query': the query that you need to perform in the given url.
                If there is no 'query' set it as open.
            'url': the base url of a site.
        """
        import multion

        # Only create new session once and continue using update session
        if self.new_session_count < 2:
            response = multion.new_session({"input": query, "url": url})
            self.new_session_count += 1
            self.tabId = response["tabId"]
            return response["message"]
        else:
            return "Continue using update session"

    def update_session(self, query: str, url: str) -> str:
        """Updates the existing browser session.

        Updates with given action and url, used consequently to handle browser
        activities after creating one session of browser.

        Args:
            'query': the query that you need to perform in the given url.
                If there is no 'query' set it as open.
            'url': the base url of a site.
        """
        import multion

        response = multion.update_session(self.tabId, {"input": query, "url": url})
        return response["message"]


class MultionClientAPIWrapper(BaseModel):
    """Wrapper for Multion Client API.

    In order to set this up, follow instructions at:
    NEED TO ADD
    """

    client: Any = MultionAPI()

    def run(self, task: str, url: str, tabId: Optional[Any]) -> str:
        """Run body through Multion Client and respond with action.

        Args:
            task:
            url:
            tabId:
        """
        if self.client.tabId is None or tabId is None:
            self.client = MultionAPI()
            message = self.client.create_session(task, url)
        else:
            message = self.client.update_session(task, url)

        return message
