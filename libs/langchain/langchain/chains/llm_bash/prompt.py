# flake8: noqa
from __future__ import annotations

import re
from typing import List

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser, OutputParserException

_PROMPT_TEMPLATE = """Если кто-то просит тебя выполнить задачу, твоя задача - придумать серию команд bash, которые выполнит эту задачу. Нет необходимости добавлять "#!/bin/bash" в свой ответ. Обязательно объясняй шаг за шагом, используя этот формат:

Question: "скопировать файлы из директории с именем 'target' в новую директорию на том же уровне, что и target, под названием 'myNewDirectory'"

Мне нужно выполнить следующие действия:
- Перечислить все файлы в директории
- Создать новую директорию
- Скопировать файлы из первой директории во вторую
```bash
ls
mkdir myNewDirectory
cp -r target/* myNewDirectory
```

Вот и весь формат. Начинай!

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
