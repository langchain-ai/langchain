# flake8: noqa

STRUCTURED_FORMAT_INSTRUCTIONS = """The output should be a markdown code snippet formatted in the following schema:

```json
{{
{format}
}}
```"""

PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below. For example, the object {{"foo": ["bar", "baz"]}} conforms to the schema {{"foo": {{"description": "a list of strings field", "type": "string"}}}}.

Here is the output schema:
```
{schema}
```"""
