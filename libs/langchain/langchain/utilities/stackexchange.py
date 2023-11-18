from stackapi import StackAPI
import html
from typing import Any, Dict, Optional

class StackExchangeAPIWrapper:
    """Wrapper for Stack Exchange API."""

    stackapi_client: Any  #: :meta private:

    def __init__(self):
        self.validate_environment()

        self.stackapi_client = StackAPI('stackoverflow')

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the required Python package exists."""
        try:
            import stackapi
            values["stackapi"] = stackapi
        except ImportError:
            raise ImportError(
                "The 'stackapi' Python package is not installed. "
                "Please install it with `pip install stackapi`."
            )
        return values

    def run(self, title: str) -> str:
        """Run query through StackExchange API and parse results."""
        output = self.stackapi_client.fetch('search/excerpts', title=title)

        result_text = ""
        for ans in output['items']:
            if ans['item_type'] == 'question':
                result_text += f"Title: {ans['title']}\n"
                if ans['answer_count'] > 0:
                    index = output['items'].index(ans)
                    result_text += f"Answer: {html.unescape(output['items'][index + 1]['excerpt'])}\n"
                result_text += "\n"

        if not result_text:
            result_text = f"No relevant results found for '{title}' on Stack Overflow"

        return result_text
