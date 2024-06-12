import re
import sys
from pathlib import Path
from typing import Union

CURR_DIR = Path(__file__).parent.absolute()

CHAT_MODEL_HEADERS = (
    "## Overview",
    "### Integration details",
    "### Model features",
    "## Setup",
    "## Instantiation",
    "## Invocation",
    "## Chaining",
    "## API reference",
)
CHAT_MODEL_REGEX = r".*".join(CHAT_MODEL_HEADERS)


def check_chat_model(path: Path) -> None:
    with open(path, "r") as f:
        doc = f.read()
    if not re.search(CHAT_MODEL_REGEX, doc, re.DOTALL):
        raise ValueError(
            f"Document {path} does not match the ChatModel Integration page template. "
            f"Please see https://github.com/langchain-ai/langchain/issues/22296 for "
            f"instructions on how to correctly format a ChatModel Integration page."
        )


def main(*new_doc_paths: Union[str, Path]) -> None:
    for path in new_doc_paths:
        path = Path(path).resolve().absolute()
        if CURR_DIR.parent / "docs" / "integrations" / "chat" in path.parents:
            print(f"Checking chat model page {path}")
            check_chat_model(path)
        else:
            continue


if __name__ == "__main__":
    main(*sys.argv[1:])
