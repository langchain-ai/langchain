from __future__ import annotations

from typing import Any, Optional, cast

from langchain_core.messages import content as types


def _convert_annotation_from_v1(annotation: types.Annotation) -> dict[str, Any]:
    """Right-inverse of _convert_citation_to_v1."""
    if annotation["type"] == "non_standard_annotation":
        return annotation["value"]

    if annotation["type"] == "citation":
        if "url" in annotation:
            # web_search_result_location
            out: dict[str, Any] = {"type": "web_search_result_location"}
            for field in ["url", "cited_text", "title"]:
                if value := annotation.get(field):
                    out[field] = value

            for key, value in annotation.get("extras", {}).items():
                out[key] = value  # noqa: PERF403

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
        # elif block["type"] == "tool_call":
        #     new_block = {"type": "function_call", "call_id": block["id"]}
        #     if "extras" in block and "item_id" in block["extras"]:
        #         new_block["id"] = block["extras"]["item_id"]
        #     if "name" in block:
        #         new_block["name"] = block["name"]
        #     if "extras" in block and "arguments" in block["extras"]:
        #         new_block["arguments"] = block["extras"]["arguments"]
        #     if any(key not in block for key in ("name", "arguments")):
        #         matching_tool_calls = [
        #             call for call in tool_calls if call["id"] == block["id"]
        #         ]
        #         if matching_tool_calls:
        #             tool_call = matching_tool_calls[0]
        #             if "name" not in block:
        #                 new_block["name"] = tool_call["name"]
        #             if "arguments" not in block:
        #                 new_block["arguments"] = json.dumps(tool_call["args"])
        #     new_content.append(new_block)

        elif (
            block["type"] == "non_standard"
            and "value" in block
            and model_provider == "anthropic"
        ):
            new_content.append(block["value"])
        else:
            new_content.append(block)

    return new_content
