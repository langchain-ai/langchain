from __future__ import annotations

import re
from typing import List

from langchain.schema import BaseOutputParser, OutputParserException


class BashOutputParser(BaseOutputParser):
    """Parser for bash output."""

    def parse(self, text: str) -> List[str]:
        if "```bash" in text:
            return self.get_code_blocks(text)
        else:
            raise OutputParserException(
                f"Failed to parse bash output. Got: {text}",
            )

    @staticmethod
    def get_code_blocks(t: str) -> List[str]:
        """Get multiple code blocks from the LLM result."""
        code_blocks: List[str] = []
        # Bash markdown code blocks
        pattern = re.compile(r"```bash(.*?)(?:\n\s*)```", re.DOTALL)
        for match in pattern.finditer(t):
            matched = match.group(1).strip()
            if matched:
                code_blocks.extend(
                    [line for line in matched.split("\n") if line.strip()]
                )

        return code_blocks

    @property
    def _type(self) -> str:
        return "bash"
