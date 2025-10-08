from langchain_core.prompts.string import mustache_schema


def test_mustache_schema_parent_child() -> None:
    template = "{{x.y}} {{x}}"
    expected = {
        "$defs": {
            "x": {
                "properties": {"y": {"default": None, "title": "Y", "type": "string"}},
                "title": "x",
                "type": "object",
            }
        },
        "properties": {"x": {"$ref": "#/$defs/x", "default": None}},
        "title": "PromptInput",
        "type": "object",
    }
    actual = mustache_schema(template).model_json_schema()
    assert expected == actual
