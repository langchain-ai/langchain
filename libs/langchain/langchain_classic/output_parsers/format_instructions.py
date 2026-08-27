STRUCTURED_FORMAT_INSTRUCTIONS = """The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{{
{format}
}}
```"""  # noqa: E501

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
```"""  # noqa: E501

YAML_FORMAT_INSTRUCTIONS = """The output should be formatted as a YAML instance that conforms to the given JSON schema below.

# Examples
## Schema
```
{{"title": "Players", "description": "A list of players", "type": "array", "items": {{"$ref": "#/definitions/Player"}}, "definitions": {{"Player": {{"title": "Player", "type": "object", "properties": {{"name": {{"title": "Name", "description": "Player name", "type": "string"}}, "avg": {{"title": "Avg", "description": "Batting average", "type": "number"}}}}, "required": ["name", "avg"]}}}}}}
```
## Well formatted instance
```
- name: John Doe
  avg: 0.3
- name: Jane Maxfield
  avg: 1.4
```

## Schema
```
{{"properties": {{"habit": {{ "description": "A common daily habit", "type": "string" }}, "sustainable_alternative": {{ "description": "An environmentally friendly alternative to the habit", "type": "string"}}}}, "required": ["habit", "sustainable_alternative"]}}
```
## Well formatted instance
```
habit: Using disposable water bottles for daily hydration.
sustainable_alternative: Switch to a reusable water bottle to reduce plastic waste and decrease your environmental footprint.
```

Please follow the standard YAML formatting conventions with an indent of 2 spaces and make sure that the data types adhere strictly to the following JSON schema:
```
{schema}
```

Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output!"""  # noqa: E501


PANDAS_DATAFRAME_FORMAT_INSTRUCTIONS = """The output should be formatted as a string as the operation, followed by a colon, followed by the column or row to be queried on, followed by optional array parameters.
1. The column names are limited to the possible columns below.
2. Arrays must either be a comma-separated list of numbers formatted as [1,3,5], or it must be in range of numbers formatted as [0..4].
3. Remember that arrays are optional and not necessarily required.
4. If the column is not in the possible columns or the operation is not a valid Pandas DataFrame operation, return why it is invalid as a sentence starting with either "Invalid column" or "Invalid operation".

As an example, for the formats:
1. String "column:num_legs" is a well-formatted instance which gets the column num_legs, where num_legs is a possible column.
2. String "row:1" is a well-formatted instance which gets row 1.
3. String "column:num_legs[1,2]" is a well-formatted instance which gets the column num_legs for rows 1 and 2, where num_legs is a possible column.
4. String "row:1[num_legs]" is a well-formatted instance which gets row 1, but for just column num_legs, where num_legs is a possible column.
5. String "mean:num_legs[1..3]" is a well-formatted instance which takes the mean of num_legs from rows 1 to 3, where num_legs is a possible column and mean is a valid Pandas DataFrame operation.
6. String "do_something:num_legs" is a badly-formatted instance, where do_something is not a valid Pandas DataFrame operation.
7. String "mean:invalid_col" is a badly-formatted instance, where invalid_col is not a possible column.

Here are the possible columns:
```
{columns}
```
"""  # noqa: E501
