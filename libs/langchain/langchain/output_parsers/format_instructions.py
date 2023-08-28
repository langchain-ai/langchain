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
"""


PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""


XML_FORMAT_INSTRUCTIONS = """The output should be formatted as a XML file.
Always mention the encoding of the file in the first line.
Output should conform to the tags below.

As an example, for the tags ["foo", "bar", "baz"], 
the string "<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>" is a well-formatted instance of the schema. 
The string "<foo>\n   <bar>\n   </foo>" is not well-formatted.
The string "<foo>\n   <tag>\n   </tag>\n</foo>" is not well-formatted.

Here are the output tags:
```
{tags}
```"""
