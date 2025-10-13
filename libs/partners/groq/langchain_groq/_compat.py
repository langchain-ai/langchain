from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.messages import content as types


def _convert_from_v1_to_groq(
    content: list[types.ContentBlock],
    model_provider: str | None,
) -> tuple[list[dict[str, Any]], dict]:
    new_content: list = []
    new_additional_kwargs: dict = {}
    for i, block in enumerate(content):
        if block["type"] == "text":
            new_content.append({"text": block.get("text", ""), "type": "text"})

        elif (
            block["type"] == "reasoning"
            and (reasoning := block.get("reasoning"))
            and model_provider == "groq"
        ):
            new_additional_kwargs["reasoning_content"] = reasoning

        elif block["type"] == "server_tool_call" and model_provider == "groq":
            new_block = {}
            if "args" in block:
                new_block["arguments"] = json.dumps(block["args"])
            if idx := block.get("extras", {}).get("index"):
                new_block["index"] = idx
            if block.get("name") == "web_search":
                new_block["type"] = "search"
            else:
                new_block["type"] = ""

            if i < len(content) - 1 and content[i + 1]["type"] == "server_tool_result":
                result = cast("types.ServerToolResult", content[i + 1])
                for k, v in result.get("extras", {}).items():
                    new_block[k] = v
                if "output" in result:
                    new_block["output"] = result["output"]

                if "executed_tools" not in new_additional_kwargs:
                    new_additional_kwargs["executed_tools"] = []
                new_additional_kwargs["executed_tools"].append(new_block)

        elif (
            block["type"] == "non_standard"
            and "value" in block
            and model_provider == "anthropic"
        ):
            new_content.append(block["value"])
        else:
            new_content.append(block)

    return new_content, new_additional_kwargs
