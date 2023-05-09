from typing import Generic, Iterable, TypeVar

from langchain.schema import BaseOutputParser

T = TypeVar("T")


class ItemParsedListOutputParser(BaseOutputParser[list[T]], Generic[T]):
    item_parser: BaseOutputParser[T]
    item_name = "item"
    separator = "\n"

    def parse(self, response: str) -> Iterable[T]:
        items = response.split(self.separator)
        items = [item.strip() for item in items]
        items = [item for item in items if item != ""]
        if self.item_parser is None:
            return items
        else:
            return [self.item_parser.parse(item) for item in items]

    def get_format_instructions(self) -> str:
        if self.item_parser is None:
            return f'Write each {self.item_name} separated by a single "{self.separator}". Do not include any other text in your answer.'
        else:
            return f'Write each {self.item_name} separated by a single "{self.separator}". Each {self.item_name} should be formatted as follows:\n{self.item_parser.get_format_instructions()}'
