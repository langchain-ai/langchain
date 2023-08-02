
from typing import List
from langchain.utilities.redis import TokenEscaper

class RedisFilter:
    escaper = TokenEscaper()

    def __init__(self, field):
        self._field = field
        self._filters = []

    def __str__(self):
        base = self.to_string()
        if self._filters:
            base += "".join(self._filters)
        return base

    def __iadd__(self, other) -> "RedisFilter":
        "intersection '+='"
        self._filters.append(f" {other.to_string()}")
        return self

    def __iand__(self, other) -> "RedisFilter":
        "union '&='"
        self._filters.append(f" | {other.to_string()}")
        return self

    def __isub__(self, other) -> "RedisFilter":
        "subtract '-='"
        self._filters.append(f" -{other.to_string()}")
        return self

    def __ixor__(self, other) -> "RedisFilter":
        "With optional '^='"
        self._filters.append(f" ~{other.to_string()}")
        return self

    def to_string(self) -> str:
        raise NotImplementedError


class TagFilter(RedisFilter):
    def __init__(self, field, tags: List[str]):
        super().__init__(field)
        self.tags = tags

    def to_string(self) -> str:
        """Converts the tag filter to a string.

        Returns:
            str: The tag filter as a string.
        """
        if not isinstance(self.tags, list):
            self.tags = [self.tags]
        return (
            "@"
            + self._field
            + ":{"
            + " | ".join([self.escaper.escape(tag) for tag in self.tags])
            + "}"
        )


class NumericFilter(RedisFilter):
    def __init__(self, field, minval, maxval, min_exclusive=False, max_exclusive=False):
        """Filter for Numeric fields.

        Args:
            field (str): The field to filter on.
            minval (int): The minimum value.
            maxval (int): The maximum value.
            min_exclusive (bool, optional): Whether the minimum value is exclusive. Defaults to False.
            max_exclusive (bool, optional): Whether the maximum value is exclusive. Defaults to False.
        """
        self.top = maxval if not max_exclusive else f"({maxval}"
        self.bottom = minval if not min_exclusive else f"({minval}"
        super().__init__(field)

    def to_string(self):
        return "@" + self._field + ":[" + str(self.bottom) + " " + str(self.top) + "]"


class TextFilter(RedisFilter):
    def __init__(self, field, text: str):
        """Filter for Text fields.
        Args:
            field (str): The field to filter on.
            text (str): The text to filter on.
        """
        super().__init__(field)
        self.text = text

    def to_string(self) -> str:
        """Converts the filter to a string.

        Returns:
            str: The filter as a string.
        """
        return "@" + self._field + ":" + self.escaper.escape(self.text)