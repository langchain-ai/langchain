# flake8: noqa

STRUCTURED_FORMAT_INSTRUCTIONS = """The output should be a markdown code snippet formatted in the following schema:

```json
{{
{format}
}}
```"""

PYDANTIC_FORMAT_INSTRUCTIONS = """The following is a JSON Schema object.

------------
{schema}
-------------

Please respond with a instance that can be validated against this JSON Schema."""

