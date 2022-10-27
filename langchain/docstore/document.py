"""Interface for interacting with a document."""
from typing import List

from pydantic import BaseModel


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    lookup_str: str = ""
    lookup_index = 0

    @property
    def paragraphs(self) -> List[str]:
        """Paragraphs of the page."""
        return self.page_content.split("\n\n")

    @property
    def summary(self) -> str:
        """Summary of the page (the first paragraph)."""
        return self.paragraphs[0]

    def lookup(self, string: str) -> str:
        """Lookup a term in the page, imitating cmd-F functionality."""
        if string.lower() != self.lookup_str:
            self.lookup_str = string.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self.paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"
