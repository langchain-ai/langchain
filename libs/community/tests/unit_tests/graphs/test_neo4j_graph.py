from langchain_community.graphs.neo4j_graph import value_sanitize


def test_value_sanitize_with_small_list():  # type: ignore[no-untyped-def]
    small_list = list(range(15))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "small_list": small_list}
    expected_output = {"key1": "value1", "small_list": small_list}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_oversized_list():  # type: ignore[no-untyped-def]
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": oversized_list}
    expected_output = {
        "key1": "value1"
        # oversized_list should not be included
    }
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_nested_oversized_list():  # type: ignore[no-untyped-def]
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": {"key": oversized_list}}
    expected_output = {"key1": "value1", "oversized_list": {}}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_dict_in_list():  # type: ignore[no-untyped-def]
    oversized_list = list(range(150))  # list size > LIST_LIMIT
    input_dict = {"key1": "value1", "oversized_list": [1, 2, {"key": oversized_list}]}
    expected_output = {"key1": "value1", "oversized_list": [1, 2, {}]}
    assert value_sanitize(input_dict) == expected_output


def test_value_sanitize_with_dict_in_nested_list():  # type: ignore[no-untyped-def]
    input_dict = {
        "key1": "value1",
        "deeply_nested_lists": [[[[{"final_nested_key": list(range(200))}]]]],
    }
    expected_output = {"key1": "value1", "deeply_nested_lists": [[[[{}]]]]}
    assert value_sanitize(input_dict) == expected_output
