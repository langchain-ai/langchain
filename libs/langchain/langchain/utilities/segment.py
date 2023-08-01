"""
Utility to collect analytics using Segment.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


class SegmentAPIWrapper(BaseModel):
    """Collect analytics using Segment.

    To use, you should have the ``segment-analytics-python`` python package installed,
    and the environment variable ``SEGMENT_WRITE_KEY``, or pass `write_key` as a named
    parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.utilities.segment import SegmentAPIWrapper segment = SegmentAPIWrapper(write_key="xxx")
            segment.run('Reported', 'user_id', dict(question='Who is the president of the United States?',
            answer='Robert Downy Jr', correct=False))
    """

    client: Any  #: :meta private:
    write_key: Optional[str] = None
    """Segment write key."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import segment.analytics as analytics
        except ImportError:
            raise ImportError(
                "Could not import segment-analytics-python python package. "
                "Please install it with `pip install segment-analytics-python`."
            )
        write_key = get_from_dict_or_env(values, "write_key", "SEGMENT_WRITE_KEY")
        analytics.write_key = write_key
        values["client"] = analytics
        return values

    def identify(self, user_id: str, traits: Optional[Dict] = None) -> None:
        """Identify a user.

        Args:
            user_id: The user's unique id.
            traits: The user's traits.
        """
        ack = self.client.identify(user_id, traits)
        return ack

    def group(self, user_id: str, group_id: str, traits: Optional[Dict] = None) -> None:
        """Group a user.

        Args:
            user_id: The user's unique id.
            group_id: The group's unique id.
            traits: The group's traits.
        """
        ack = self.client.group(user_id, group_id, traits)
        return ack

    def page(self, user_id: str, name: str, properties: Optional[Dict] = None) -> None:
        """Send a page event to Segment.

        Args:
            user_id: The user's unique id.
            name: The page name.
            properties: The page's properties.
        """
        ack = self.client.page(user_id, name, properties)
        return ack

    def screen(
        self, user_id: str, name: str, properties: Optional[Dict] = None
    ) -> None:
        """Send a screen event to Segment.

        Args:
            user_id: The user's unique id.
            name: The screen name.
            properties: The screen's properties.
        """
        ack = self.client.screen(user_id, name, properties)
        return ack

    def alias(self, user_id: str, previous_id: str) -> None:
        """Send an alias event to Segment.

        Args:
            user_id: The user's unique id.
            previous_id: The user's previous unique id.
        """
        ack = self.client.alias(user_id, previous_id)
        return ack

    def run(self, event: str, user_id: str, properties: Optional[Dict] = None) -> None:
        """Send an event to Segment.

        Args:
            event: The event name.
            user_id: The user's unique id.
            properties: The event's properties.
        """
        ack = self.client.track(user_id, event, properties)
        return ack
