from datetime import datetime, timedelta, timezone

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.utils import comma_list


class DatetimeOutputParser(BaseOutputParser[datetime]):
    """Parse the output of an LLM call to a datetime."""

    format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
    """The string value that is used as the datetime format.

    Update this to match the desired datetime format for your application.
    """

    def get_format_instructions(self) -> str:
        """Returns the format instructions for the given format."""
        if self.format == "%Y-%m-%dT%H:%M:%S.%fZ":
            examples = comma_list(
                [
                    "2023-07-04T14:30:00.000000Z",
                    "1999-12-31T23:59:59.999999Z",
                    "2025-01-01T00:00:00.000000Z",
                ],
            )
        else:
            try:
                now = datetime.now(tz=timezone.utc)
                examples = comma_list(
                    [
                        now.strftime(self.format),
                        (now.replace(year=now.year - 1)).strftime(self.format),
                        (now - timedelta(days=1)).strftime(self.format),
                    ],
                )
            except ValueError:
                # Fallback if the format is very unusual
                examples = f"e.g., a valid string in the format {self.format}"

        return (
            f"Write a datetime string that matches the "
            f"following pattern: '{self.format}'.\n\n"
            f"Examples: {examples}\n\n"
            f"Return ONLY this string, no other words!"
        )

    def parse(self, response: str) -> datetime:
        """Parse a string into a datetime object."""
        try:
            return datetime.strptime(response.strip(), self.format)  # noqa: DTZ007
        except ValueError as e:
            msg = f"Could not parse datetime string: {response}"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
        return "datetime"
