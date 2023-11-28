import html
from typing import Any, Dict

from langchain.pydantic_v1 import BaseModel, root_validator


class StackExchangeAPIWrapper(BaseModel):
    """Wrapper for Stack Exchange API."""

    stackapi_client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the required Python package exists."""
        try:
            from stackapi import StackAPI

            values["stackapi_client"] = StackAPI("stackoverflow")
        except ImportError:
            raise ImportError(
                "The 'stackapi' Python package is not installed. "
                "Please install it with `pip install stackapi`."
            )
        return values

    def run(self, title: str) -> str:
        """Run query through StackExchange API and parse results."""
        SITE = self.stackapi_client
        output = SITE.fetch("search/excerpts", title=title)
        if(len(output["items"]) < 1):
             return f"No relevant results found for '{title}' on Stack Overflow"
        res_text = ""
        count = 0
        for sol in output["items"]:
            if(count > 3):
                break
            if sol["item_type"] == "question":
                res_text += f"Title: {sol['title']}\n"
                if sol["answer_count"] > 0:
                    index = output["items"].index(sol)
                    res_text += "Answer: "
                    res_text += (
                        f"{html.unescape(output['items'][index + 1]['excerpt'])}"
                    )
                    res_text += "\n"
                res_text += "\n"
                count += 1

        return res_text
