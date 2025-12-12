from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.messages import content as types


def _convert_annotation_from_v1(annotation: types.Annotation) -> dict[str, Any]:
    """Convert LangChain annotation format to Anthropic's native citation format."""
    if annotation["type"] == "non_standard_annotation":
        return annotation["value"]

    if annotation["type"] == "citation":
        if "url" in annotation:
            # web_search_result_location
            out: dict[str, Any] = {}
            if cited_text := annotation.get("cited_text"):
                out["cited_text"] = cited_text
            if "encrypted_index" in annotation.get("extras", {}):
                out["encrypted_index"] = annotation.get("extras", {})["encrypted_index"]
            if "title" in annotation:
                out["title"] = annotation["title"]
            out["type"] = "web_search_result_location"
            out["url"] = annotation.get("url")

            for key, value in annotation.get("extras", {}).items():
                if key not in out:
                    out[key] = value

            return out

        if "start_char_index" in annotation.get("extras", {}):
            # char_location
            out = {"type": "char_location"}
            for field in ["cited_text"]:
                if value := annotation.get(field):
                    out[field] = value
            if title := annotation.get("title"):
                out["document_title"] = title

            for key, value in annotation.get("extras", {}).items():
                out[key] = value
            out = {k: out[k] for k in sorted(out)}

            return out

        if "search_result_index" in annotation.get("extras", {}):
            # search_result_location
            out = {"type": "search_result_location"}
            for field in ["cited_text", "title"]:
                if value := annotation.get(field):
                    out[field] = value

            for key, value in annotation.get("extras", {}).items():
                out[key] = value

            return out

        if "start_block_index" in annotation.get("extras", {}):
            # content_block_location
            out = {}
            if cited_text := annotation.get("cited_text"):
                out["cited_text"] = cited_text
            if "document_index" in annotation.get("extras", {}):
                out["document_index"] = annotation.get("extras", {})["document_index"]
            if "title" in annotation:
                out["document_title"] = annotation["title"]

            for key, value in annotation.get("extras", {}).items():
                if key not in out:
                    out[key] = value

            out["type"] = "content_block_location"
            return out

        if "start_page_number" in annotation.get("extras", {}):
            # page_location
            out = {"type": "page_location"}
            for field in ["cited_text"]:
                if value := annotation.get(field):
                    out[field] = value
            if title := annotation.get("title"):
                out["document_title"] = title

            for key, value in annotation.get("extras", {}).items():
                out[key] = value

            return out

        return cast(dict[str, Any], annotation)

    return cast(dict[str, Any], annotation)


def _convert_from_v1_to_anthropic(
    content: list[types.ContentBlock],
    tool_calls: list[types.ToolCall],
    model_provider: str | None,
) -> list[dict[str, Any]]:
    new_content: list = []
    for block in content:
        if block["type"] == "text":
            if model_provider == "anthropic" and "annotations" in block:
                new_block: dict[str, Any] = {"type": "text"}
                new_block["citations"] = [
                    _convert_annotation_from_v1(a) for a in block["annotations"]
                ]
                if "text" in block:
                    new_block["text"] = block["text"]
            else:
                new_block = {"text": block.get("text", ""), "type": "text"}
            new_content.append(new_block)

        elif block["type"] == "tool_call":
            tool_use_block = {
                "type": "tool_use",
                "name": block.get("name", ""),
                "input": block.get("args", {}),
                "id": block.get("id", ""),
            }
            if "caller" in block.get("extras", {}):
                tool_use_block["caller"] = block["extras"]["caller"]
            new_content.append(tool_use_block)

        elif block["type"] == "tool_call_chunk":
            if isinstance(block["args"], str):
                try:
                    input_ = json.loads(block["args"] or "{}")
                except json.JSONDecodeError:
                    input_ = {}
            else:
                input_ = block.get("args") or {}
            new_content.append(
                {
                    "type": "tool_use",
                    "name": block.get("name", ""),
                    "input": input_,
                    "id": block.get("id", ""),
                }
            )

        elif block["type"] == "reasoning" and model_provider == "anthropic":
            new_block = {}
            if "reasoning" in block:
                new_block["thinking"] = block["reasoning"]
            new_block["type"] = "thinking"
            if signature := block.get("extras", {}).get("signature"):
                new_block["signature"] = signature

            new_content.append(new_block)

        elif block["type"] == "server_tool_call" and model_provider == "anthropic":
            new_block = {}
            if "id" in block:
                new_block["id"] = block["id"]
            new_block["input"] = block.get("args", {})
            if partial_json := block.get("extras", {}).get("partial_json"):
                new_block["input"] = {}
                new_block["partial_json"] = partial_json
            else:
                pass
            if block.get("name") == "code_interpreter":
                new_block["name"] = "code_execution"
            elif block.get("name") == "remote_mcp":
                if "tool_name" in block.get("extras", {}):
                    new_block["name"] = block["extras"]["tool_name"]
                if "server_name" in block.get("extras", {}):
                    new_block["server_name"] = block["extras"]["server_name"]
            else:
                new_block["name"] = block.get("name", "")
            if block.get("name") == "remote_mcp":
                new_block["type"] = "mcp_tool_use"
            else:
                new_block["type"] = "server_tool_use"
            new_content.append(new_block)

        elif block["type"] == "server_tool_result" and model_provider == "anthropic":
            new_block = {}
            if "output" in block:
                new_block["content"] = block["output"]
            server_tool_result_type = block.get("extras", {}).get("block_type", "")
            if server_tool_result_type == "mcp_tool_result":
                new_block["is_error"] = block.get("status") == "error"
            if "tool_call_id" in block:
                new_block["tool_use_id"] = block["tool_call_id"]
            new_block["type"] = server_tool_result_type
            new_content.append(new_block)

        elif (
            block["type"] == "non_standard"
            and "value" in block
            and model_provider == "anthropic"
        ):
            new_content.append(block["value"])
        else:
            new_content.append(block)

    return new_content
