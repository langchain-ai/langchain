"""Core encoding logic for TOON format."""

from __future__ import annotations

from typing import Literal, cast

from .constants import (
    LIST_ITEM_MARKER,
    LIST_ITEM_PREFIX,
    DelimiterType,
    Depth,
    JsonArray,
    JsonObject,
    JsonPrimitive,
    JsonValue,
)
from .formatters import (
    encode_key,
    encode_primitive,
    format_header,
    join_encoded_values,
)
from .normalize import (
    is_array_of_arrays,
    is_array_of_objects,
    is_array_of_primitives,
    is_json_array,
    is_json_object,
    is_json_primitive,
)
from .writer import LineWriter


def encode_value(
    value: JsonValue,
    *,
    indent: int = 2,
    delimiter: DelimiterType = ",",
    length_marker: Literal["#", False] = False,
) -> str:
    """Encode a JSON value to TOON format.

    Args:
        value: JSON-compatible value to encode.
        indent: Spaces per indentation level.
        delimiter: Delimiter character for arrays and tables.
        length_marker: If `'#'`, prefix array lengths with `'#'`.

    Returns:
        TOON-formatted string.
    """
    if is_json_primitive(value):
        return encode_primitive(cast("JsonPrimitive", value), delimiter)

    writer = LineWriter(indent)

    if is_json_array(value):
        encode_array(
            None, cast("JsonArray", value), writer, 0, indent, delimiter, length_marker
        )
    elif is_json_object(value):
        encode_object(
            cast("JsonObject", value), writer, 0, indent, delimiter, length_marker
        )

    return writer.to_string()


def encode_object(
    value: JsonObject,
    writer: LineWriter,
    depth: Depth,
    indent: int,
    delimiter: DelimiterType,
    length_marker: Literal["#", False],
) -> None:
    """Encode an object as key-value pairs.

    Args:
        value: Object to encode.
        writer: Line writer for output.
        depth: Current indentation depth.
        indent: Spaces per indentation level.
        delimiter: Delimiter character.
        length_marker: Length marker flag.
    """
    keys = list(value.keys())
    for key in keys:
        encode_key_value_pair(
            key, value[key], writer, depth, indent, delimiter, length_marker
        )


def encode_key_value_pair(
    key: str,
    value: JsonValue,
    writer: LineWriter,
    depth: Depth,
    indent: int,
    delimiter: DelimiterType,
    length_marker: Literal["#", False],
) -> None:
    """Encode a single key-value pair.

    Args:
        key: Object key.
        value: Value to encode.
        writer: Line writer for output.
        depth: Current indentation depth.
        indent: Spaces per indentation level.
        delimiter: Delimiter character.
        length_marker: Length marker flag.
    """
    encoded_key = encode_key(key)

    if is_json_primitive(value):
        encoded_value = encode_primitive(cast("JsonPrimitive", value), delimiter)
        writer.push(depth, f"{encoded_key}: {encoded_value}")
    elif is_json_array(value):
        encode_array(
            key,
            cast("JsonArray", value),
            writer,
            depth,
            indent,
            delimiter,
            length_marker,
        )
    elif is_json_object(value):
        obj = cast("JsonObject", value)
        nested_keys = list(obj.keys())
        if not nested_keys:
            writer.push(depth, f"{encoded_key}:")
        else:
            writer.push(depth, f"{encoded_key}:")
            encode_object(obj, writer, depth + 1, indent, delimiter, length_marker)


def encode_array(
    key: str | None,
    value: JsonArray,
    writer: LineWriter,
    depth: Depth,
    indent: int,
    delimiter: DelimiterType,
    length_marker: Literal["#", False],
) -> None:
    """Encode an array, choosing the appropriate format.

    Arrays are encoded differently based on their content:
    - Empty arrays: `key[0]:`
    - Primitive arrays: Inline format `key[N]: val1,val2,val3`
    - Tabular arrays: Objects with identical keys rendered as rows
    - Nested arrays: Arrays of primitive arrays as list items
    - Mixed arrays: General list item format with `-` markers

    Args:
        key: Optional key name for the array.
        value: Array to encode.
        writer: Line writer for output.
        depth: Current indentation depth.
        indent: Spaces per indentation level.
        delimiter: Delimiter character.
        length_marker: Length marker flag.
    """
    items = list(value)

    if not items:
        header_str = format_header(
            0, key=key, delimiter=delimiter, length_marker=length_marker
        )
        writer.push(depth, header_str)
        return

    if is_array_of_primitives(items):
        encode_inline_primitive_array(
            key,
            cast("list[JsonPrimitive]", items),
            writer,
            depth,
            delimiter,
            length_marker,
        )
        return

    if is_array_of_arrays(items):
        # At this point we know all items are arrays
        arr_items = cast("list[JsonArray]", items)
        if all(is_array_of_primitives(inner) for inner in arr_items):
            encode_array_of_arrays_as_list_items(
                key,
                arr_items,
                writer,
                depth,
                delimiter,
                length_marker,
            )
            return

    if is_array_of_objects(items):
        obj_items = cast("list[JsonObject]", items)
        header: list[str] | None = detect_tabular_header(obj_items)
        if header is not None:
            encode_array_of_objects_as_tabular(
                key, obj_items, header, writer, depth, delimiter, length_marker
            )
        else:
            encode_mixed_array_as_list_items(
                key,
                cast("list[JsonValue]", obj_items),
                writer,
                depth,
                indent,
                delimiter,
                length_marker,
            )
        return

    encode_mixed_array_as_list_items(
        key, items, writer, depth, indent, delimiter, length_marker
    )


def encode_inline_primitive_array(
    prefix: str | None,
    values: list[JsonPrimitive],
    writer: LineWriter,
    depth: Depth,
    delimiter: DelimiterType,
    length_marker: Literal["#", False],
) -> None:
    """Encode an array of primitives in inline format.

    Args:
        prefix: Optional key name.
        values: List of primitive values.
        writer: Line writer for output.
        depth: Current indentation depth.
        delimiter: Delimiter character.
        length_marker: Length marker flag.
    """
    formatted = format_inline_array(values, delimiter, prefix, length_marker)
    writer.push(depth, formatted)


def encode_array_of_arrays_as_list_items(
    prefix: str | None,
    values: list[JsonArray],
    writer: LineWriter,
    depth: Depth,
    delimiter: DelimiterType,
    length_marker: Literal["#", False],
) -> None:
    """Encode arrays of primitive arrays as nested list items.

    Args:
        prefix: Optional key name.
        values: List of arrays.
        writer: Line writer for output.
        depth: Current indentation depth.
        delimiter: Delimiter character.
        length_marker: Length marker flag.
    """
    header = format_header(
        len(values),
        key=prefix,
        delimiter=delimiter,
        length_marker=length_marker,
    )
    writer.push(depth, header)

    for arr in values:
        if is_array_of_primitives(arr):
            inline = format_inline_array(
                cast("list[JsonPrimitive]", list(arr)),
                delimiter,
                None,
                length_marker,
            )
            writer.push(depth + 1, f"{LIST_ITEM_PREFIX}{inline}")


def format_inline_array(
    values: list[JsonPrimitive],
    delimiter: DelimiterType,
    prefix: str | None = None,
    length_marker: Literal["#", False] = False,
) -> str:
    """Format an array of primitives as a single inline string.

    Args:
        values: List of primitive values.
        delimiter: Delimiter character.
        prefix: Optional key name.
        length_marker: Length marker flag.

    Returns:
        Formatted inline array string.
    """
    header = format_header(
        len(values),
        key=prefix,
        delimiter=delimiter,
        length_marker=length_marker,
    )
    if not values:
        return header

    joined_value = join_encoded_values(values, delimiter)
    return f"{header} {joined_value}"


def encode_array_of_objects_as_tabular(
    prefix: str | None,
    rows: list[JsonObject],
    header: list[str],
    writer: LineWriter,
    depth: Depth,
    delimiter: DelimiterType,
    length_marker: Literal["#", False],
) -> None:
    """Encode an array of objects in tabular format.

    Args:
        prefix: Optional key name.
        rows: List of objects with identical keys.
        header: List of field names.
        writer: Line writer for output.
        depth: Current indentation depth.
        delimiter: Delimiter character.
        length_marker: Length marker flag.
    """
    header_str = format_header(
        len(rows),
        key=prefix,
        fields=header,
        delimiter=delimiter,
        length_marker=length_marker,
    )
    writer.push(depth, header_str)
    write_tabular_rows(rows, header, writer, depth + 1, delimiter)


def detect_tabular_header(rows: list[JsonObject]) -> list[str] | None:
    """Detect if an array of objects can be rendered in tabular format.

    Checks if all objects have identical keys and all values are primitives.

    Args:
        rows: List of objects to check.

    Returns:
        List of field names if tabular format is possible, `None` otherwise.
    """
    if not rows:
        return None

    first_row = rows[0]
    first_keys = list(first_row.keys())
    if not first_keys:
        return None

    if is_tabular_array(rows, first_keys):
        return first_keys
    return None


def is_tabular_array(rows: list[JsonObject], header: list[str]) -> bool:
    """Check if all rows have identical keys with primitive values.

    Args:
        rows: List of objects to check.
        header: Expected list of keys.

    Returns:
        `True` if all rows match the tabular format.
    """
    for row in rows:
        keys = list(row.keys())
        if len(keys) != len(header):
            return False

        for key in header:
            if key not in row:
                return False
            if not is_json_primitive(row[key]):
                return False
    return True


def write_tabular_rows(
    rows: list[JsonObject],
    header: list[str],
    writer: LineWriter,
    depth: Depth,
    delimiter: DelimiterType,
) -> None:
    """Write object rows in tabular format.

    Args:
        rows: List of objects.
        header: List of field names.
        writer: Line writer for output.
        depth: Current indentation depth.
        delimiter: Delimiter character.
    """
    for row in rows:
        values = [row[key] for key in header]
        primitives = cast("list[JsonPrimitive]", values)
        joined_value = join_encoded_values(primitives, delimiter)
        writer.push(depth, joined_value)


def encode_mixed_array_as_list_items(
    prefix: str | None,
    items: list[JsonValue],
    writer: LineWriter,
    depth: Depth,
    indent: int,
    delimiter: DelimiterType,
    length_marker: Literal["#", False],
) -> None:
    """Encode a mixed array as list items with `-` markers.

    Args:
        prefix: Optional key name.
        items: List of mixed values.
        writer: Line writer for output.
        depth: Current indentation depth.
        indent: Spaces per indentation level.
        delimiter: Delimiter character.
        length_marker: Length marker flag.
    """
    header = format_header(
        len(items),
        key=prefix,
        delimiter=delimiter,
        length_marker=length_marker,
    )
    writer.push(depth, header)

    for item in items:
        if is_json_primitive(item):
            prim = encode_primitive(cast("JsonPrimitive", item), delimiter)
            writer.push(depth + 1, f"{LIST_ITEM_PREFIX}{prim}")
        elif is_json_array(item):
            arr = cast("JsonArray", item)
            if is_array_of_primitives(arr):
                inline = format_inline_array(
                    cast("list[JsonPrimitive]", list(arr)),
                    delimiter,
                    None,
                    length_marker,
                )
                writer.push(depth + 1, f"{LIST_ITEM_PREFIX}{inline}")
            else:
                # Nested complex arrays
                encode_array(
                    None, arr, writer, depth + 1, indent, delimiter, length_marker
                )
        elif is_json_object(item):
            encode_object_as_list_item(
                cast("JsonObject", item),
                writer,
                depth + 1,
                indent,
                delimiter,
                length_marker,
            )


def encode_object_as_list_item(
    obj: JsonObject,
    writer: LineWriter,
    depth: Depth,
    indent: int,
    delimiter: DelimiterType,
    length_marker: Literal["#", False],
) -> None:
    """Encode an object as a list item with `-` marker.

    Args:
        obj: Object to encode.
        writer: Line writer for output.
        depth: Current indentation depth.
        indent: Spaces per indentation level.
        delimiter: Delimiter character.
        length_marker: Length marker flag.
    """
    keys = list(obj.keys())
    if not keys:
        writer.push(depth, LIST_ITEM_MARKER)
        return

    first_key = keys[0]
    encoded_key = encode_key(first_key)
    first_value = obj[first_key]

    if is_json_primitive(first_value):
        encoded_value = encode_primitive(cast("JsonPrimitive", first_value), delimiter)
        writer.push(
            depth,
            f"{LIST_ITEM_PREFIX}{encoded_key}: {encoded_value}",
        )
    elif is_json_array(first_value):
        arr = cast("JsonArray", first_value)
        if is_array_of_primitives(arr):
            formatted = format_inline_array(
                cast("list[JsonPrimitive]", list(arr)),
                delimiter,
                first_key,
                length_marker,
            )
            writer.push(depth, f"{LIST_ITEM_PREFIX}{formatted}")
        elif is_array_of_objects(arr):
            obj_list = cast("list[JsonObject]", list(arr))
            header: list[str] | None = detect_tabular_header(obj_list)
            if header is not None:
                header_str = format_header(
                    len(arr),
                    key=first_key,
                    fields=header,
                    delimiter=delimiter,
                    length_marker=length_marker,
                )
                writer.push(depth, f"{LIST_ITEM_PREFIX}{header_str}")
                write_tabular_rows(
                    obj_list,
                    header,
                    writer,
                    depth + 1,
                    delimiter,
                )
            else:
                array_len = len(arr)
                writer.push(depth, f"{LIST_ITEM_PREFIX}{encoded_key}[{array_len}]:")
                for item in arr:
                    encode_object_as_list_item(
                        cast("JsonObject", item),
                        writer,
                        depth + 1,
                        indent,
                        delimiter,
                        length_marker,
                    )
        else:
            writer.push(depth, f"{LIST_ITEM_PREFIX}{encoded_key}[{len(arr)}]:")
            for item in arr:
                if is_json_primitive(item):
                    prim = encode_primitive(cast("JsonPrimitive", item), delimiter)
                    writer.push(depth + 1, f"{LIST_ITEM_PREFIX}{prim}")
                elif is_json_array(item):
                    inner_arr = cast("JsonArray", item)
                    if is_array_of_primitives(inner_arr):
                        inline = format_inline_array(
                            cast("list[JsonPrimitive]", list(inner_arr)),
                            delimiter,
                            None,
                            length_marker,
                        )
                        writer.push(depth + 1, f"{LIST_ITEM_PREFIX}{inline}")
                elif is_json_object(item):
                    encode_object_as_list_item(
                        cast("JsonObject", item),
                        writer,
                        depth + 1,
                        indent,
                        delimiter,
                        length_marker,
                    )
    elif is_json_object(first_value):
        nested_obj = cast("JsonObject", first_value)
        nested_keys = list(nested_obj.keys())
        writer.push(depth, f"{LIST_ITEM_PREFIX}{encoded_key}:")
        if nested_keys:
            encode_object(
                nested_obj, writer, depth + 2, indent, delimiter, length_marker
            )

    # Encode remaining keys
    for key in keys[1:]:
        encode_key_value_pair(
            key, obj[key], writer, depth + 1, indent, delimiter, length_marker
        )
