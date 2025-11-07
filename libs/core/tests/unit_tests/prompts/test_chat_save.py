import json
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import pytest
import yaml

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate


@pytest.mark.parametrize(
    ("suffix", "loader"),
    [
        (".json", json.load),
        (".yaml", yaml.safe_load),
    ],
)
def test_chat_prompt_template_save_roundtrip(
    tmp_path: Path, suffix: str, loader: Callable[[IO[str]], Any]
) -> None:
    prompt = ChatPromptTemplate.from_messages(
        [HumanMessagePromptTemplate.from_template("Hello {name}")]
    )

    out = tmp_path / f"prompt{suffix}"
    prompt.save(out)

    assert out.exists()

    with out.open("r", encoding="utf-8") as f:
        data = loader(f)

    # minimal structural assertions
    assert data["_type"] == "chat"
    # messages should have exactly one human template entry
    assert isinstance(data["messages"], list)
    assert len(data["messages"]) == 1
    first = data["messages"][0]
    # Accept various serializer shapes; only assert that an entry exists and is structured
    assert isinstance(first, (dict, list))
