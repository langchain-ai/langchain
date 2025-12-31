import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_core.documents import Document
from langchain_core.load import InitValidator, Serializable, dumpd, dumps, load, loads
from langchain_core.load.serializable import _is_field_useful
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)


class NonBoolObj:
    def __bool__(self) -> bool:
        msg = "Truthiness can't be determined"
        raise ValueError(msg)

    def __eq__(self, other: object) -> bool:
        msg = "Equality can't be determined"
        raise ValueError(msg)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__

    __hash__ = None  # type: ignore[assignment]


def test_simple_serialization() -> None:
    class Foo(Serializable):
        bar: int
        baz: str

    foo = Foo(bar=1, baz="hello")
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "lc": 1,
        "repr": "Foo(bar=1, baz='hello')",
        "type": "not_implemented",
    }


def test_simple_serialization_is_serializable() -> None:
    class Foo(Serializable):
        bar: int
        baz: str

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }


def test_simple_serialization_secret() -> None:
    """Test handling of secrets."""

    class Foo(Serializable):
        bar: int
        baz: str
        secret: SecretStr
        secret_2: str

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_secrets(self) -> dict[str, str]:
            return {"secret": "MASKED_SECRET", "secret_2": "MASKED_SECRET_2"}

    foo = Foo(
        bar=1, baz="baz", secret=SecretStr("SUPER_SECRET"), secret_2="SUPER_SECRET"
    )
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {
            "bar": 1,
            "baz": "baz",
            "secret": {"id": ["MASKED_SECRET"], "lc": 1, "type": "secret"},
            "secret_2": {"id": ["MASKED_SECRET_2"], "lc": 1, "type": "secret"},
        },
        "lc": 1,
        "type": "constructor",
    }


def test__is_field_useful() -> None:
    class ArrayObj:
        def __bool__(self) -> bool:
            msg = "Truthiness can't be determined"
            raise ValueError(msg)

        def __eq__(self, other: object) -> bool:
            return self  # type: ignore[return-value]

        __hash__ = None  # type: ignore[assignment]

    default_x = ArrayObj()
    default_y = NonBoolObj()

    class Foo(Serializable):
        x: ArrayObj = Field(default=default_x)
        y: NonBoolObj = Field(default=default_y)
        # Make sure works for fields without default.
        z: ArrayObj

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
        )

    foo = Foo(x=ArrayObj(), y=NonBoolObj(), z=ArrayObj())
    assert _is_field_useful(foo, "x", foo.x)
    assert _is_field_useful(foo, "y", foo.y)

    foo = Foo(x=default_x, y=default_y, z=ArrayObj())
    assert not _is_field_useful(foo, "x", foo.x)
    assert not _is_field_useful(foo, "y", foo.y)


class Foo(Serializable):
    bar: int
    baz: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_simple_deserialization() -> None:
    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    serialized_foo = dumpd(foo)
    assert serialized_foo == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }
    new_foo = load(serialized_foo, allowed_objects=[Foo], valid_namespaces=["tests"])
    assert new_foo == foo


def test_disallowed_deserialization() -> None:
    foo = Foo(bar=1, baz="hello")
    serialized_foo = dumpd(foo)
    with pytest.raises(ValueError, match="not allowed"):
        load(serialized_foo, allowed_objects=[], valid_namespaces=["tests"])


class Foo2(Serializable):
    bar: int
    baz: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_simple_deserialization_with_additional_imports() -> None:
    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    serialized_foo = dumpd(foo)
    assert serialized_foo == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }
    new_foo = load(
        serialized_foo,
        allowed_objects=[Foo2],
        valid_namespaces=["tests"],
        additional_import_mappings={
            ("tests", "unit_tests", "load", "test_serializable", "Foo"): (
                "tests",
                "unit_tests",
                "load",
                "test_serializable",
                "Foo2",
            )
        },
    )
    assert isinstance(new_foo, Foo2)


class Foo3(Serializable):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    non_bool: NonBoolObj

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_repr() -> None:
    foo = Foo3(
        content="repr",
        non_bool=NonBoolObj(),
    )
    assert repr(foo) == "Foo3(content='repr', non_bool=NonBoolObj)"


def test_str() -> None:
    foo = Foo3(
        content="str",
        non_bool=NonBoolObj(),
    )
    assert str(foo) == "content='str' non_bool=NonBoolObj"


def test_serialization_with_pydantic() -> None:
    class MyModel(BaseModel):
        x: int
        y: str

    my_model = MyModel(x=1, y="hello")
    llm_response = ChatGeneration(
        message=AIMessage(
            content='{"x": 1, "y": "hello"}', additional_kwargs={"parsed": my_model}
        )
    )
    ser = dumpd(llm_response)
    deser = load(ser, allowed_objects=[ChatGeneration, AIMessage])
    assert isinstance(deser, ChatGeneration)
    assert deser.message.content
    assert deser.message.additional_kwargs["parsed"] == my_model.model_dump()


def test_serialization_with_generation() -> None:
    generation = Generation(text="hello-world")
    assert dumpd(generation)["kwargs"] == {"text": "hello-world", "type": "Generation"}


def test_serialization_with_ignore_unserializable_fields() -> None:
    data = {
        "messages": [
            [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "AIMessage"],
                    "kwargs": {
                        "content": "Call tools to get entity details",
                        "response_metadata": {
                            "other_field": "foo",
                            "create_date": {
                                "lc": 1,
                                "type": "not_implemented",
                                "id": ["datetime", "datetime"],
                                "repr": "datetime.datetime(2025, 7, 15, 13, 14, 0, 000000, tzinfo=datetime.timezone.utc)",  # noqa: E501
                            },
                        },
                        "type": "ai",
                        "id": "00000000-0000-0000-0000-000000000000",
                    },
                },
            ]
        ]
    }
    # Load directly (no dumpd - this is already serialized data)
    deser = load(data, allowed_objects=[AIMessage], ignore_unserializable_fields=True)
    assert deser == {
        "messages": [
            [
                AIMessage(
                    id="00000000-0000-0000-0000-000000000000",
                    content="Call tools to get entity details",
                    response_metadata={
                        "other_field": "foo",
                        "create_date": None,
                    },
                )
            ]
        ]
    }


# Tests for dumps() function
def test_dumps_basic_serialization() -> None:
    """Test basic string serialization with `dumps()`."""
    foo = Foo(bar=42, baz="test")
    json_str = dumps(foo)

    # Should be valid JSON
    parsed = json.loads(json_str)
    assert parsed == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 42, "baz": "test"},
        "lc": 1,
        "type": "constructor",
    }


def test_dumps_pretty_formatting() -> None:
    """Test pretty printing functionality."""
    foo = Foo(bar=1, baz="hello")

    # Test pretty=True with default indent
    pretty_json = dumps(foo, pretty=True)
    assert "  " in pretty_json

    # Test custom indent (4-space)
    custom_indent = dumps(foo, pretty=True, indent=4)
    assert "    " in custom_indent

    # Verify it's still valid JSON
    parsed = json.loads(pretty_json)
    assert parsed["kwargs"]["bar"] == 1


def test_dumps_invalid_default_kwarg() -> None:
    """Test that passing `'default'` as kwarg raises ValueError."""
    foo = Foo(bar=1, baz="test")

    with pytest.raises(ValueError, match="`default` should not be passed to dumps"):
        dumps(foo, default=lambda x: x)


def test_dumps_additional_json_kwargs() -> None:
    """Test that additional JSON kwargs are passed through."""
    foo = Foo(bar=1, baz="test")

    compact_json = dumps(foo, separators=(",", ":"))
    assert ", " not in compact_json  # Should be compact

    # Test sort_keys
    sorted_json = dumps(foo, sort_keys=True)
    parsed = json.loads(sorted_json)
    assert parsed == dumpd(foo)


def test_dumps_non_serializable_object() -> None:
    """Test `dumps()` behavior with non-serializable objects."""

    class NonSerializable:
        def __init__(self, value: int) -> None:
            self.value = value

    obj = NonSerializable(42)
    json_str = dumps(obj)

    # Should create a "not_implemented" representation
    parsed = json.loads(json_str)
    assert parsed["lc"] == 1
    assert parsed["type"] == "not_implemented"
    assert "NonSerializable" in parsed["repr"]


def test_dumps_mixed_data_structure() -> None:
    """Test `dumps()` with complex nested data structures."""
    data = {
        "serializable": Foo(bar=1, baz="test"),
        "list": [1, 2, {"nested": "value"}],
        "primitive": "string",
    }

    json_str = dumps(data)
    parsed = json.loads(json_str)

    # Serializable object should be properly serialized
    assert parsed["serializable"]["type"] == "constructor"
    # Primitives should remain unchanged
    assert parsed["list"] == [1, 2, {"nested": "value"}]
    assert parsed["primitive"] == "string"


def test_document_normal_metadata_allowed() -> None:
    """Test that `Document` metadata without `'lc'` key works fine."""
    doc = Document(
        page_content="Hello world",
        metadata={"source": "test.txt", "page": 1, "nested": {"key": "value"}},
    )
    serialized = dumpd(doc)

    loaded = load(serialized, allowed_objects=[Document])
    assert loaded.page_content == "Hello world"

    expected = {"source": "test.txt", "page": 1, "nested": {"key": "value"}}
    assert loaded.metadata == expected


class TestEscaping:
    """Tests that escape-based serialization prevents injection attacks.

    When user data contains an `'lc'` key, it's escaped during serialization
    (wrapped in `{"__lc_escaped__": ...}`). During deserialization, escaped
    dicts are unwrapped and returned as plain dicts - NOT instantiated as
    LC objects.
    """

    def test_document_metadata_with_lc_key_escaped(self) -> None:
        """Test that `Document` metadata with `'lc'` key round-trips as plain dict."""
        # User data that looks like an LC constructor - should be escaped, not executed
        suspicious_metadata = {"lc": 1, "type": "constructor", "id": ["some", "module"]}
        doc = Document(page_content="test", metadata=suspicious_metadata)

        # Serialize - should escape the metadata
        serialized = dumpd(doc)
        assert serialized["kwargs"]["metadata"] == {
            "__lc_escaped__": suspicious_metadata
        }

        # Deserialize - should restore original metadata as plain dict
        loaded = load(serialized, allowed_objects=[Document])
        assert loaded.metadata == suspicious_metadata  # Plain dict, not instantiated

    def test_document_metadata_with_nested_lc_key_escaped(self) -> None:
        """Test that nested `'lc'` key in `Document` metadata is escaped."""
        suspicious_nested = {"lc": 1, "type": "constructor", "id": ["some", "module"]}
        doc = Document(page_content="test", metadata={"nested": suspicious_nested})

        serialized = dumpd(doc)
        # The nested dict with 'lc' key should be escaped
        assert serialized["kwargs"]["metadata"]["nested"] == {
            "__lc_escaped__": suspicious_nested
        }

        loaded = load(serialized, allowed_objects=[Document])
        assert loaded.metadata == {"nested": suspicious_nested}

    def test_document_metadata_with_lc_key_in_list_escaped(self) -> None:
        """Test that `'lc'` key in list items within `Document` metadata is escaped."""
        suspicious_item = {"lc": 1, "type": "constructor", "id": ["some", "module"]}
        doc = Document(page_content="test", metadata={"items": [suspicious_item]})

        serialized = dumpd(doc)
        assert serialized["kwargs"]["metadata"]["items"][0] == {
            "__lc_escaped__": suspicious_item
        }

        loaded = load(serialized, allowed_objects=[Document])
        assert loaded.metadata == {"items": [suspicious_item]}

    def test_malicious_payload_not_instantiated(self) -> None:
        """Test that malicious LC-like structures in user data are NOT instantiated."""
        # An attacker might craft a payload with a valid AIMessage structure in metadata
        malicious_data = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "document", "Document"],
            "kwargs": {
                "page_content": "test",
                "metadata": {
                    # This looks like a valid LC object but is in escaped form
                    "__lc_escaped__": {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain_core", "messages", "ai", "AIMessage"],
                        "kwargs": {"content": "injected message"},
                    }
                },
            },
        }

        # Even though AIMessage is allowed, the metadata should remain as dict
        loaded = load(malicious_data, allowed_objects=[Document, AIMessage])
        assert loaded.page_content == "test"
        # The metadata is the original dict (unescaped), NOT an AIMessage instance
        assert loaded.metadata == {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_core", "messages", "ai", "AIMessage"],
            "kwargs": {"content": "injected message"},
        }
        assert not isinstance(loaded.metadata, AIMessage)

    def test_message_additional_kwargs_with_lc_key_escaped(self) -> None:
        """Test that `AIMessage` `additional_kwargs` with `'lc'` is escaped."""
        suspicious_data = {"lc": 1, "type": "constructor", "id": ["x", "y"]}
        msg = AIMessage(
            content="Hello",
            additional_kwargs={"data": suspicious_data},
        )

        serialized = dumpd(msg)
        assert serialized["kwargs"]["additional_kwargs"]["data"] == {
            "__lc_escaped__": suspicious_data
        }

        loaded = load(serialized, allowed_objects=[AIMessage])
        assert loaded.additional_kwargs == {"data": suspicious_data}

    def test_message_response_metadata_with_lc_key_escaped(self) -> None:
        """Test that `AIMessage` `response_metadata` with `'lc'` is escaped."""
        suspicious_data = {"lc": 1, "type": "constructor", "id": ["x", "y"]}
        msg = AIMessage(content="Hello", response_metadata=suspicious_data)

        serialized = dumpd(msg)
        assert serialized["kwargs"]["response_metadata"] == {
            "__lc_escaped__": suspicious_data
        }

        loaded = load(serialized, allowed_objects=[AIMessage])
        assert loaded.response_metadata == suspicious_data

    def test_double_escape_handling(self) -> None:
        """Test that data containing escape key itself is properly handled."""
        # User data that contains our escape key
        data_with_escape_key = {"__lc_escaped__": "some_value"}
        doc = Document(page_content="test", metadata=data_with_escape_key)

        serialized = dumpd(doc)
        # Should be double-escaped since it looks like an escaped dict
        assert serialized["kwargs"]["metadata"] == {
            "__lc_escaped__": {"__lc_escaped__": "some_value"}
        }

        loaded = load(serialized, allowed_objects=[Document])
        assert loaded.metadata == {"__lc_escaped__": "some_value"}


class TestDumpdEscapesLcKeyInPlainDicts:
    """Tests that `dumpd()` escapes `'lc'` keys in plain dict kwargs."""

    def test_normal_message_not_escaped(self) -> None:
        """Test that normal `AIMessage` without `'lc'` key is not escaped."""
        msg = AIMessage(
            content="Hello",
            additional_kwargs={"tool_calls": []},
            response_metadata={"model": "gpt-4"},
        )
        serialized = dumpd(msg)
        assert serialized["kwargs"]["content"] == "Hello"
        # No escape wrappers for normal data
        assert "__lc_escaped__" not in str(serialized)

    def test_document_metadata_with_lc_key_escaped(self) -> None:
        """Test that `Document` with `'lc'` key in metadata is escaped."""
        doc = Document(
            page_content="test",
            metadata={"lc": 1, "type": "constructor"},
        )

        serialized = dumpd(doc)
        # Should be escaped, not blocked
        assert serialized["kwargs"]["metadata"] == {
            "__lc_escaped__": {"lc": 1, "type": "constructor"}
        }

    def test_document_metadata_with_nested_lc_key_escaped(self) -> None:
        """Test that `Document` with nested `'lc'` in metadata is escaped."""
        doc = Document(
            page_content="test",
            metadata={"nested": {"lc": 1}},
        )

        serialized = dumpd(doc)
        assert serialized["kwargs"]["metadata"]["nested"] == {
            "__lc_escaped__": {"lc": 1}
        }

    def test_message_additional_kwargs_with_lc_key_escaped(self) -> None:
        """Test `AIMessage` with `'lc'` in `additional_kwargs` is escaped."""
        msg = AIMessage(
            content="Hello",
            additional_kwargs={"malicious": {"lc": 1}},
        )

        serialized = dumpd(msg)
        assert serialized["kwargs"]["additional_kwargs"]["malicious"] == {
            "__lc_escaped__": {"lc": 1}
        }

    def test_message_response_metadata_with_lc_key_escaped(self) -> None:
        """Test `AIMessage` with `'lc'` in `response_metadata` is escaped."""
        msg = AIMessage(
            content="Hello",
            response_metadata={"lc": 1},
        )

        serialized = dumpd(msg)
        assert serialized["kwargs"]["response_metadata"] == {
            "__lc_escaped__": {"lc": 1}
        }


class TestInitValidator:
    """Tests for `init_validator` on `load()` and `loads()`."""

    def test_init_validator_allows_valid_kwargs(self) -> None:
        """Test that `init_validator` returning None allows deserialization."""
        msg = AIMessage(content="Hello")
        serialized = dumpd(msg)

        def allow_all(_class_path: tuple[str, ...], _kwargs: dict[str, Any]) -> None:
            pass  # Allow all by doing nothing

        loaded = load(serialized, allowed_objects=[AIMessage], init_validator=allow_all)
        assert loaded == msg

    def test_init_validator_blocks_deserialization(self) -> None:
        """Test that `init_validator` can block deserialization by raising."""
        doc = Document(page_content="test", metadata={"source": "test.txt"})
        serialized = dumpd(doc)

        def block_metadata(
            _class_path: tuple[str, ...], kwargs: dict[str, Any]
        ) -> None:
            if "metadata" in kwargs:
                msg = "Metadata not allowed"
                raise ValueError(msg)

        with pytest.raises(ValueError, match="Metadata not allowed"):
            load(serialized, allowed_objects=[Document], init_validator=block_metadata)

    def test_init_validator_receives_correct_class_path(self) -> None:
        """Test that `init_validator` receives the correct class path."""
        msg = AIMessage(content="Hello")
        serialized = dumpd(msg)

        received_class_paths: list[tuple[str, ...]] = []

        def capture_class_path(
            class_path: tuple[str, ...], _kwargs: dict[str, Any]
        ) -> None:
            received_class_paths.append(class_path)

        load(serialized, allowed_objects=[AIMessage], init_validator=capture_class_path)

        assert len(received_class_paths) == 1
        assert received_class_paths[0] == (
            "langchain",
            "schema",
            "messages",
            "AIMessage",
        )

    def test_init_validator_receives_correct_kwargs(self) -> None:
        """Test that `init_validator` receives the kwargs dict."""
        msg = AIMessage(content="Hello world", name="test_name")
        serialized = dumpd(msg)

        received_kwargs: list[dict[str, Any]] = []

        def capture_kwargs(
            _class_path: tuple[str, ...], kwargs: dict[str, Any]
        ) -> None:
            received_kwargs.append(kwargs)

        load(serialized, allowed_objects=[AIMessage], init_validator=capture_kwargs)

        assert len(received_kwargs) == 1
        assert "content" in received_kwargs[0]
        assert received_kwargs[0]["content"] == "Hello world"
        assert "name" in received_kwargs[0]
        assert received_kwargs[0]["name"] == "test_name"

    def test_init_validator_with_loads(self) -> None:
        """Test that `init_validator` works with `loads()` function."""
        doc = Document(page_content="test", metadata={"key": "value"})
        json_str = dumps(doc)

        def block_metadata(
            _class_path: tuple[str, ...], kwargs: dict[str, Any]
        ) -> None:
            if "metadata" in kwargs:
                msg = "Metadata not allowed"
                raise ValueError(msg)

        with pytest.raises(ValueError, match="Metadata not allowed"):
            loads(json_str, allowed_objects=[Document], init_validator=block_metadata)

    def test_init_validator_none_allows_all(self) -> None:
        """Test that `init_validator=None` (default) allows all kwargs."""
        msg = AIMessage(content="Hello")
        serialized = dumpd(msg)

        # Should work without init_validator
        loaded = load(serialized, allowed_objects=[AIMessage])
        assert loaded == msg

    def test_init_validator_type_alias_exists(self) -> None:
        """Test that `InitValidator` type alias is exported and usable."""

        def my_validator(_class_path: tuple[str, ...], _kwargs: dict[str, Any]) -> None:
            pass

        validator_typed: InitValidator = my_validator
        assert callable(validator_typed)

    def test_init_validator_blocks_specific_class(self) -> None:
        """Test blocking deserialization for a specific class."""
        doc = Document(page_content="test", metadata={"source": "test.txt"})
        serialized = dumpd(doc)

        def block_documents(
            class_path: tuple[str, ...], _kwargs: dict[str, Any]
        ) -> None:
            if class_path == ("langchain", "schema", "document", "Document"):
                msg = "Documents not allowed"
                raise ValueError(msg)

        with pytest.raises(ValueError, match="Documents not allowed"):
            load(serialized, allowed_objects=[Document], init_validator=block_documents)


class TestJinja2SecurityBlocking:
    """Tests blocking Jinja2 templates by default."""

    def test_fstring_template_allowed(self) -> None:
        """Test that f-string templates deserialize successfully."""
        # Serialized ChatPromptTemplate with f-string format
        serialized = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "prompts", "chat", "ChatPromptTemplate"],
            "kwargs": {
                "input_variables": ["name"],
                "messages": [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": [
                            "langchain",
                            "prompts",
                            "chat",
                            "HumanMessagePromptTemplate",
                        ],
                        "kwargs": {
                            "prompt": {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["name"],
                                    "template": "Hello {name}",
                                    "template_format": "f-string",
                                },
                            }
                        },
                    }
                ],
            },
        }

        # f-string should deserialize successfully
        loaded = load(
            serialized,
            allowed_objects=[
                ChatPromptTemplate,
                HumanMessagePromptTemplate,
                PromptTemplate,
            ],
        )
        assert isinstance(loaded, ChatPromptTemplate)
        assert loaded.input_variables == ["name"]

    def test_jinja2_template_blocked(self) -> None:
        """Test that Jinja2 templates are blocked by default."""
        # Malicious serialized payload attempting to use jinja2
        malicious_serialized = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "prompts", "chat", "ChatPromptTemplate"],
            "kwargs": {
                "input_variables": ["name"],
                "messages": [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": [
                            "langchain",
                            "prompts",
                            "chat",
                            "HumanMessagePromptTemplate",
                        ],
                        "kwargs": {
                            "prompt": {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["name"],
                                    "template": "{{ name }}",
                                    "template_format": "jinja2",
                                },
                            }
                        },
                    }
                ],
            },
        }

        # jinja2 should be blocked by default
        with pytest.raises(ValueError, match="Jinja2 templates are not allowed"):
            load(
                malicious_serialized,
                allowed_objects=[
                    ChatPromptTemplate,
                    HumanMessagePromptTemplate,
                    PromptTemplate,
                ],
            )

    def test_jinja2_blocked_standalone_prompt_template(self) -> None:
        """Test blocking Jinja2 on standalone `PromptTemplate`."""
        serialized_jinja2 = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["name"],
                "template": "{{ name }}",
                "template_format": "jinja2",
            },
        }

        serialized_fstring = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["name"],
                "template": "{name}",
                "template_format": "f-string",
            },
        }

        # f-string should work
        loaded = load(
            serialized_fstring,
            allowed_objects=[PromptTemplate],
        )
        assert isinstance(loaded, PromptTemplate)
        assert loaded.template == "{name}"

        # jinja2 should be blocked by default
        with pytest.raises(ValueError, match="Jinja2 templates are not allowed"):
            load(
                serialized_jinja2,
                allowed_objects=[PromptTemplate],
            )

    def test_jinja2_blocked_by_default(self) -> None:
        """Test that Jinja2 templates are blocked by default."""
        serialized_jinja2 = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["name"],
                "template": "{{ name }}",
                "template_format": "jinja2",
            },
        }

        serialized_fstring = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["name"],
                "template": "{name}",
                "template_format": "f-string",
            },
        }

        # f-string should work
        loaded = load(serialized_fstring, allowed_objects=[PromptTemplate])
        assert isinstance(loaded, PromptTemplate)
        assert loaded.template == "{name}"

        # jinja2 should be blocked by default
        with pytest.raises(ValueError, match="Jinja2 templates are not allowed"):
            load(serialized_jinja2, allowed_objects=[PromptTemplate])


class TestClassSpecificValidatorsInLoad:
    """Tests that load() properly integrates with class-specific validators."""

    def test_class_validator_registry_exists(self) -> None:
        """Test that the CLASS_INIT_VALIDATORS registry is accessible."""
        from langchain_core.load.validators import CLASS_INIT_VALIDATORS

        # Registry should exist and have Bedrock entries
        assert isinstance(CLASS_INIT_VALIDATORS, dict)
        assert len(CLASS_INIT_VALIDATORS) > 0

    def test_init_validator_called_when_no_class_validator(self) -> None:
        """Test that init_validator is called if no class-specific validator."""
        msg = AIMessage(content="test")
        serialized = dumpd(msg)

        init_validator_called = []

        def custom_init_validator(
            _class_path: tuple[str, ...], _kwargs: dict[str, Any]
        ) -> None:
            init_validator_called.append(True)

        # Should successfully deserialize and call init_validator
        loaded = load(
            serialized,
            allowed_objects=[AIMessage],
            init_validator=custom_init_validator
        )
        assert loaded == msg
        assert len(init_validator_called) == 1


class TestBedrockValidators:
    """Tests for Bedrock SSRF protection validator."""

    def test_bedrock_validator_blocks_endpoint_url(self) -> None:
        """Test that _bedrock_validator blocks `endpoint_url` parameter."""
        from langchain_core.load.validators import _bedrock_validator

        class_path = ("langchain", "llms", "bedrock", "BedrockLLM")
        kwargs = {
            "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "region_name": "us-west-2",
            "endpoint_url": "http://169.254.169.254/latest/meta-data",
        }

        with pytest.raises(ValueError, match="endpoint_url"):
            _bedrock_validator(class_path, kwargs)

        with pytest.raises(ValueError, match="SSRF"):
            _bedrock_validator(class_path, kwargs)

    def test_bedrock_validator_blocks_base_url(self) -> None:
        """Test that _bedrock_validator blocks `base_url` parameter."""
        from langchain_core.load.validators import _bedrock_validator

        class_path = ("langchain_aws", "chat_models", "ChatBedrockConverse")
        kwargs = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "region_name": "us-west-2",
            "base_url": "http://malicious-site.com",
        }

        with pytest.raises(ValueError, match="base_url"):
            _bedrock_validator(class_path, kwargs)

        with pytest.raises(ValueError, match="SSRF"):
            _bedrock_validator(class_path, kwargs)

    def test_bedrock_validator_blocks_both_parameters(self) -> None:
        """Test that _bedrock_validator blocks when both params are present."""
        from langchain_core.load.validators import _bedrock_validator

        class_path = ("langchain", "chat_models", "bedrock", "ChatBedrock")
        kwargs = {
            "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "region_name": "us-west-2",
            "endpoint_url": "http://attacker.com",
            "base_url": "http://another-attacker.com",
        }

        with pytest.raises(ValueError) as exc_info:
            _bedrock_validator(class_path, kwargs)

        error_msg = str(exc_info.value)

        assert "endpoint_url" in error_msg or "base_url" in error_msg
        assert "SSRF" in error_msg

    def test_bedrock_validator_allows_safe_parameters(self) -> None:
        """Test that _bedrock_validator allows safe parameters through."""
        from langchain_core.load.validators import _bedrock_validator

        class_path = ("langchain", "llms", "bedrock", "Bedrock")
        kwargs = {
            "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "region_name": "us-west-2",
            "credentials_profile_name": "default",
            "streaming": True,
            "model_kwargs": {"temperature": 0.7},
        }

        _bedrock_validator(class_path, kwargs)
