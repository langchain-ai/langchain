import html
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, model_validator


class StackExchangeAPIWrapper(BaseModel):
    """Wrapper for Stack Exchange API."""

    client: Any = None  #: :meta private:
    max_results: int = 3
    """Max number of results to include in output."""
    query_type: Literal["all", "title", "body"] = "all"
    """Which part of StackOverflows items to match against. One of 'all', 'title', 
        'body'. Defaults to 'all'.
    """
    fetch_params: Dict[str, Any] = Field(default_factory=dict)
    """Additional params to pass to StackApi.fetch."""
    result_separator: str = "\n\n"
    """Separator between question,answer pairs."""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that the required Python package exists."""
        try:
            from stackapi import StackAPI

            values["client"] = StackAPI("stackoverflow")
        except ImportError:
            raise ImportError(
                "The 'stackapi' Python package is not installed. "
                "Please install it with `pip install stackapi`."
            )
        return values

    def run(self, query: str) -> str:
        """Run query through StackExchange API and parse results."""

        query_key = "q" if self.query_type == "all" else self.query_type
        output = self.client.fetch(
            "search/excerpts", **{query_key: query}, **self.fetch_params
        )
        if len(output["items"]) < 1:
            return f"No relevant results found for '{query}' on Stack Overflow."
        questions = [
            item for item in output["items"] if item["item_type"] == "question"
        ][: self.max_results]
        answers = [item for item in output["items"] if item["item_type"] == "answer"]
        results = []
        for question in questions:
            res_text = f"Question: {question['title']}\n{question['excerpt']}"
            relevant_answers = [
                answer
                for answer in answers
                if answer["question_id"] == question["question_id"]
            ]
            accepted_answers = [
                answer for answer in relevant_answers if answer["is_accepted"]
            ]
            if relevant_answers:
                top_answer = (
                    accepted_answers[0] if accepted_answers else relevant_answers[0]
                )
                excerpt = html.unescape(top_answer["excerpt"])
                res_text += f"\nAnswer: {excerpt}"
            results.append(res_text)

        return self.result_separator.join(results)
