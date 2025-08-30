from __future__ import annotations

import json
from typing import Any, Optional, cast

from langchain_core.messages import content as types


def _convert_annotation_from_v1(annotation: types.Annotation) -> dict[str, Any]:
    """Right-inverse of _convert_citation_to_v1."""
    if annotation["type"] == "non_standard_annotation":
        return annotation["value"]

    if annotation["type"] == "citation":
        if "url" in annotation:
            # web_search_result_location
            out: dict[str, Any] = {}
            if cited_text := annotation.get("cited_text"):
                out["cited_text"] = cited_text
            if "encrypted_index" in annotation.get("extras", {}):
                out["encrypted_index"] = annotation["extras"]["encrypted_index"]
            if "title" in annotation:
                out["title"] = annotation["title"]
            out["type"] = "web_search_result_location"
            if "url" in annotation:
                out["url"] = annotation["url"]

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
                out["document_index"] = annotation["extras"]["document_index"]
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
    model_provider: Optional[str],
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
            new_content.append(
                {
                    "type": "tool_use",
                    "name": block.get("name", ""),
                    "input": block.get("args", {}),
                    "id": block.get("id", ""),
                }
            )

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

        elif block["type"] == "web_search_call" and model_provider == "anthropic":
            new_block = {}
            if "id" in block:
                new_block["id"] = block["id"]

            if (query := block.get("query")) and "input" not in block:
                new_block["input"] = {"query": query}
            elif input_ := block.get("extras", {}).get("input"):
                new_block["input"] = input_
            elif partial_json := block.get("extras", {}).get("partial_json"):
                new_block["input"] = {}
                new_block["partial_json"] = partial_json
            else:
                pass
            new_block["name"] = "web_search"
            new_block["type"] = "server_tool_use"
            new_content.append(new_block)

        elif block["type"] == "web_search_result" and model_provider == "anthropic":
            new_block = {}
            if "content" in block.get("extras", {}):
                new_block["content"] = block["extras"]["content"]
            if "id" in block:
                new_block["tool_use_id"] = block["id"]
            new_block["type"] = "web_search_tool_result"
            new_content.append(new_block)

        elif block["type"] == "code_interpreter_call" and model_provider == "anthropic":
            new_block = {}
            if "id" in block:
                new_block["id"] = block["id"]
            if (code := block.get("code")) and "input" not in block:
                new_block["input"] = {"code": code}
            elif input_ := block.get("extras", {}).get("input"):
                new_block["input"] = input_
            elif partial_json := block.get("extras", {}).get("partial_json"):
                new_block["input"] = {}
                new_block["partial_json"] = partial_json
            else:
                pass
            new_block["name"] = "code_execution"
            new_block["type"] = "server_tool_use"
            new_content.append(new_block)

        elif (
            block["type"] == "code_interpreter_result" and model_provider == "anthropic"
        ):
            new_block = {}
            if (output := block.get("output", [])) and len(output) == 1:
                code_interpreter_output = output[0]
                code_execution_content = {}
                if "content" in block.get("extras", {}):
                    code_execution_content["content"] = block["extras"]["content"]
                elif (file_ids := block.get("file_ids")) and isinstance(file_ids, list):
                    code_execution_content["content"] = [
                        {"file_id": file_id, "type": "code_execution_output"}
                        for file_id in file_ids
                    ]
                else:
                    code_execution_content["content"] = []
                if "return_code" in code_interpreter_output:
                    code_execution_content["return_code"] = code_interpreter_output[
                        "return_code"
                    ]
                code_execution_content["stderr"] = code_interpreter_output.get(
                    "stderr", ""
                )
                if "stdout" in code_interpreter_output:
                    code_execution_content["stdout"] = code_interpreter_output["stdout"]
                code_execution_content["type"] = "code_execution_result"
                new_block["content"] = code_execution_content
            elif "error_code" in block.get("extras", {}):
                code_execution_content = {
                    "error_code": block["extras"]["error_code"],
                    "type": "code_execution_tool_result_error",
                }
                new_block["content"] = code_execution_content
            else:
                pass
            if "id" in block:
                new_block["tool_use_id"] = block["id"]
            new_block["type"] = "code_execution_tool_result"
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
