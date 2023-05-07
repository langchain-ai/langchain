from typing import Generic, Iterable, TypeVar

from langchain.schema import BaseOutputParser

T = TypeVar("T")


class RemoveQuotesOutputParser(BaseOutputParser[list[T]], Generic[T]):
    parser: BaseOutputParser[T] = None
    quotes: list[tuple[str, str]] = [
        ("'", "'"),
        ('"', '"'),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
        ("《", "》"),
    ]

    def parse(self, response: str) -> Iterable[T]:
        # We don't know what order quotes will enclose the response, so we need to check all of them in all orders. This should be fine because we rarely have many quotes to check.
        quotes_to_check = self.quotes.copy()
        for _ in self.quotes:
            for left, right in quotes_to_check:
                if response.startswith(left) and response.endswith(right):
                    response = response[len(left) : -len(right)]
                    quotes_to_check.remove((left, right))
                    break
            else:
                break
        if self.parser is not None:
            response = self.parser.parse(response)
        return response

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()
