# flake8: noqa
from __future__ import annotations

import re
from typing import List

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser, OutputParserException

_PROMPT_TEMPLATE = """If someone asks you to perform a task, your job is to come up with a series of bash commands that will perform the task. There is no need to put "#!/bin/bash" in your answer. Make sure to reason step by step, using this format:

Question: "copy the files in the directory named 'target' into a new directory at the same level as target called 'myNewDirectory'"

I need to take the following actions:
- List all files in the directory
- Create a new directory
- Copy the files from the first directory into the second directory
```bash
ls
mkdir myNewDirectory
cp -r target/* myNewDirectory
```

That is the format. Begin!

Question: {question}"""


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


PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
    output_parser=BashOutputParser(),
)
