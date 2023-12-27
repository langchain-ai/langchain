# flake8: noqa

STRUCTURED_FORMAT_INSTRUCTIONS = """The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{{
{format}
}}
```"""

STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS = """
```json
{{
{format}
}}
```"""


PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""

YAML_FORMAT_INSTRUCTIONS = """The output should be formatted as a YAML instance that conforms to the given JSON schema below.

As an example, for the schema
```
{{'title': 'Players', 'description': 'A list of players', 'type': 'array', 'items': {{'$ref': '#/definitions/Player'}}, 'definitions': {{'Player': {{'title': 'Player', 'type': 'object', 'properties': {{'name': {{'title': 'Name', 'description': 'Player name', 'type': 'string'}}, 'avg': {{'title': 'Avg', 'description': 'Batting average', 'type': 'number'}}}}, 'required': ['name', 'avg']}}}}}}
```
a well formatted instance would be:
```
- name: John Doe
  avg: 0.3
- name: Jane Maxfield
  avg: 1.4
```

Please follow the standard YAML formatting conventions with an indent of 2 spaces and make sure that the data types adhere strictly to the following JSON schema: 
```
{schema}
```

Make sure to always enclose the YAML output in triple backticks (```)"""
