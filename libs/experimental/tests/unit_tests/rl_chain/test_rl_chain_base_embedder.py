from typing import List, Union

import pytest
from test_utils import MockEncoder

import langchain_experimental.rl_chain.base as base

encoded_keyword = "[encoded]"


@pytest.mark.requires("vowpal_wabbit_next")
def test_simple_context_str_no_emb() -> None:
    expected = [{"a_namespace": "test"}]
    assert base.embed("test", MockEncoder(), "a_namespace") == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_simple_context_str_w_emb() -> None:
    str1 = "test"
    encoded_str1 = base.stringify_embedding(list(encoded_keyword + str1))
    expected = [{"a_namespace": encoded_str1}]
    assert base.embed(base.Embed(str1), MockEncoder(), "a_namespace") == expected
    expected_embed_and_keep = [{"a_namespace": str1 + " " + encoded_str1}]
    assert (
        base.embed(base.EmbedAndKeep(str1), MockEncoder(), "a_namespace")
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_simple_context_str_w_nested_emb() -> None:
    # nested embeddings, innermost wins
    str1 = "test"
    encoded_str1 = base.stringify_embedding(list(encoded_keyword + str1))
    expected = [{"a_namespace": encoded_str1}]
    assert (
        base.embed(base.EmbedAndKeep(base.Embed(str1)), MockEncoder(), "a_namespace")
        == expected
    )

    expected2 = [{"a_namespace": str1 + " " + encoded_str1}]
    assert (
        base.embed(base.Embed(base.EmbedAndKeep(str1)), MockEncoder(), "a_namespace")
        == expected2
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_context_w_namespace_no_emb() -> None:
    expected = [{"test_namespace": "test"}]
    assert base.embed({"test_namespace": "test"}, MockEncoder()) == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_context_w_namespace_w_emb() -> None:
    str1 = "test"
    encoded_str1 = base.stringify_embedding(list(encoded_keyword + str1))
    expected = [{"test_namespace": encoded_str1}]
    assert base.embed({"test_namespace": base.Embed(str1)}, MockEncoder()) == expected
    expected_embed_and_keep = [{"test_namespace": str1 + " " + encoded_str1}]
    assert (
        base.embed({"test_namespace": base.EmbedAndKeep(str1)}, MockEncoder())
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_context_w_namespace_w_emb2() -> None:
    str1 = "test"
    encoded_str1 = base.stringify_embedding(list(encoded_keyword + str1))
    expected = [{"test_namespace": encoded_str1}]
    assert base.embed(base.Embed({"test_namespace": str1}), MockEncoder()) == expected
    expected_embed_and_keep = [{"test_namespace": str1 + " " + encoded_str1}]
    assert (
        base.embed(base.EmbedAndKeep({"test_namespace": str1}), MockEncoder())
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_context_w_namespace_w_some_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    encoded_str2 = base.stringify_embedding(list(encoded_keyword + str2))
    expected = [{"test_namespace": str1, "test_namespace2": encoded_str2}]
    assert (
        base.embed(
            {"test_namespace": str1, "test_namespace2": base.Embed(str2)}, MockEncoder()
        )
        == expected
    )
    expected_embed_and_keep = [
        {
            "test_namespace": str1,
            "test_namespace2": str2 + " " + encoded_str2,
        }
    ]
    assert (
        base.embed(
            {"test_namespace": str1, "test_namespace2": base.EmbedAndKeep(str2)},
            MockEncoder(),
        )
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_simple_action_strlist_no_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    expected = [{"a_namespace": str1}, {"a_namespace": str2}, {"a_namespace": str3}]
    to_embed: List[Union[str, base._Embed]] = [str1, str2, str3]
    assert base.embed(to_embed, MockEncoder(), "a_namespace") == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_simple_action_strlist_w_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str1 = base.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = base.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = base.stringify_embedding(list(encoded_keyword + str3))
    expected = [
        {"a_namespace": encoded_str1},
        {"a_namespace": encoded_str2},
        {"a_namespace": encoded_str3},
    ]
    assert (
        base.embed(base.Embed([str1, str2, str3]), MockEncoder(), "a_namespace")
        == expected
    )
    expected_embed_and_keep = [
        {"a_namespace": str1 + " " + encoded_str1},
        {"a_namespace": str2 + " " + encoded_str2},
        {"a_namespace": str3 + " " + encoded_str3},
    ]
    assert (
        base.embed(base.EmbedAndKeep([str1, str2, str3]), MockEncoder(), "a_namespace")
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_simple_action_strlist_w_some_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str2 = base.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = base.stringify_embedding(list(encoded_keyword + str3))
    expected = [
        {"a_namespace": str1},
        {"a_namespace": encoded_str2},
        {"a_namespace": encoded_str3},
    ]
    assert (
        base.embed(
            [str1, base.Embed(str2), base.Embed(str3)], MockEncoder(), "a_namespace"
        )
        == expected
    )
    expected_embed_and_keep = [
        {"a_namespace": str1},
        {"a_namespace": str2 + " " + encoded_str2},
        {"a_namespace": str3 + " " + encoded_str3},
    ]
    assert (
        base.embed(
            [str1, base.EmbedAndKeep(str2), base.EmbedAndKeep(str3)],
            MockEncoder(),
            "a_namespace",
        )
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_action_w_namespace_no_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    expected = [
        {"test_namespace": str1},
        {"test_namespace": str2},
        {"test_namespace": str3},
    ]
    assert (
        base.embed(
            [
                {"test_namespace": str1},
                {"test_namespace": str2},
                {"test_namespace": str3},
            ],
            MockEncoder(),
        )
        == expected
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_action_w_namespace_w_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str1 = base.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = base.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = base.stringify_embedding(list(encoded_keyword + str3))
    expected = [
        {"test_namespace": encoded_str1},
        {"test_namespace": encoded_str2},
        {"test_namespace": encoded_str3},
    ]
    assert (
        base.embed(
            [
                {"test_namespace": base.Embed(str1)},
                {"test_namespace": base.Embed(str2)},
                {"test_namespace": base.Embed(str3)},
            ],
            MockEncoder(),
        )
        == expected
    )
    expected_embed_and_keep = [
        {"test_namespace": str1 + " " + encoded_str1},
        {"test_namespace": str2 + " " + encoded_str2},
        {"test_namespace": str3 + " " + encoded_str3},
    ]
    assert (
        base.embed(
            [
                {"test_namespace": base.EmbedAndKeep(str1)},
                {"test_namespace": base.EmbedAndKeep(str2)},
                {"test_namespace": base.EmbedAndKeep(str3)},
            ],
            MockEncoder(),
        )
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_action_w_namespace_w_emb2() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str1 = base.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = base.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = base.stringify_embedding(list(encoded_keyword + str3))
    expected = [
        {"test_namespace1": encoded_str1},
        {"test_namespace2": encoded_str2},
        {"test_namespace3": encoded_str3},
    ]
    assert (
        base.embed(
            base.Embed(
                [
                    {"test_namespace1": str1},
                    {"test_namespace2": str2},
                    {"test_namespace3": str3},
                ]
            ),
            MockEncoder(),
        )
        == expected
    )
    expected_embed_and_keep = [
        {"test_namespace1": str1 + " " + encoded_str1},
        {"test_namespace2": str2 + " " + encoded_str2},
        {"test_namespace3": str3 + " " + encoded_str3},
    ]
    assert (
        base.embed(
            base.EmbedAndKeep(
                [
                    {"test_namespace1": str1},
                    {"test_namespace2": str2},
                    {"test_namespace3": str3},
                ]
            ),
            MockEncoder(),
        )
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_action_w_namespace_w_some_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str2 = base.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = base.stringify_embedding(list(encoded_keyword + str3))
    expected = [
        {"test_namespace": str1},
        {"test_namespace": encoded_str2},
        {"test_namespace": encoded_str3},
    ]
    assert (
        base.embed(
            [
                {"test_namespace": str1},
                {"test_namespace": base.Embed(str2)},
                {"test_namespace": base.Embed(str3)},
            ],
            MockEncoder(),
        )
        == expected
    )
    expected_embed_and_keep = [
        {"test_namespace": str1},
        {"test_namespace": str2 + " " + encoded_str2},
        {"test_namespace": str3 + " " + encoded_str3},
    ]
    assert (
        base.embed(
            [
                {"test_namespace": str1},
                {"test_namespace": base.EmbedAndKeep(str2)},
                {"test_namespace": base.EmbedAndKeep(str3)},
            ],
            MockEncoder(),
        )
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_action_w_namespace_w_emb_w_more_than_one_item_in_first_dict() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str1 = base.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = base.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = base.stringify_embedding(list(encoded_keyword + str3))
    expected = [
        {"test_namespace": encoded_str1, "test_namespace2": str1},
        {"test_namespace": encoded_str2, "test_namespace2": str2},
        {"test_namespace": encoded_str3, "test_namespace2": str3},
    ]
    assert (
        base.embed(
            [
                {"test_namespace": base.Embed(str1), "test_namespace2": str1},
                {"test_namespace": base.Embed(str2), "test_namespace2": str2},
                {"test_namespace": base.Embed(str3), "test_namespace2": str3},
            ],
            MockEncoder(),
        )
        == expected
    )
    expected_embed_and_keep = [
        {
            "test_namespace": str1 + " " + encoded_str1,
            "test_namespace2": str1,
        },
        {
            "test_namespace": str2 + " " + encoded_str2,
            "test_namespace2": str2,
        },
        {
            "test_namespace": str3 + " " + encoded_str3,
            "test_namespace2": str3,
        },
    ]
    assert (
        base.embed(
            [
                {"test_namespace": base.EmbedAndKeep(str1), "test_namespace2": str1},
                {"test_namespace": base.EmbedAndKeep(str2), "test_namespace2": str2},
                {"test_namespace": base.EmbedAndKeep(str3), "test_namespace2": str3},
            ],
            MockEncoder(),
        )
        == expected_embed_and_keep
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_one_namespace_w_list_of_features_no_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    expected = [{"test_namespace": [str1, str2]}]
    assert base.embed({"test_namespace": [str1, str2]}, MockEncoder()) == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_one_namespace_w_list_of_features_w_some_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    encoded_str2 = base.stringify_embedding(list(encoded_keyword + str2))
    expected = [{"test_namespace": [str1, encoded_str2]}]
    assert (
        base.embed({"test_namespace": [str1, base.Embed(str2)]}, MockEncoder())
        == expected
    )


@pytest.mark.requires("vowpal_wabbit_next")
def test_nested_list_features_throws() -> None:
    with pytest.raises(ValueError):
        base.embed({"test_namespace": [[1, 2], [3, 4]]}, MockEncoder())


@pytest.mark.requires("vowpal_wabbit_next")
def test_dict_in_list_throws() -> None:
    with pytest.raises(ValueError):
        base.embed({"test_namespace": [{"a": 1}, {"b": 2}]}, MockEncoder())


@pytest.mark.requires("vowpal_wabbit_next")
def test_nested_dict_throws() -> None:
    with pytest.raises(ValueError):
        base.embed({"test_namespace": {"a": {"b": 1}}}, MockEncoder())


@pytest.mark.requires("vowpal_wabbit_next")
def test_list_of_tuples_throws() -> None:
    with pytest.raises(ValueError):
        base.embed({"test_namespace": [("a", 1), ("b", 2)]}, MockEncoder())
