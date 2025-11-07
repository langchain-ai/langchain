import json
from pathlib import Path

import pytest
import yaml

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate


@pytest.mark.parametrize("suffix, loader", [(".json", json.load), (".yaml", yaml.safe_load)])
def test_chat_prompt_template_save_roundtrip(tmp_path: Path, suffix: str, loader):
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
    # Accept either compact or expanded form depending on serializer
    if isinstance(first, dict) and first.get("type") == "human":
        # Expanded form
        content = first.get("content")
        assert isinstance(content, dict)
        assert content.get("template") == "Hello {name}"
    else:
        # Compact tuple-like form serialized as list [role, template]
        assert isinstance(first, list)
        assert first[0] == "human"
        assert first[1] == "Hello {name}"
