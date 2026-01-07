"""Test text splitting functionality."""

from __future__ import annotations

import random
import re
import string
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core._api import suppress_langchain_beta_warning
from langchain_core.documents import Document

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    TextSplitter,
    Tokenizer,
)
from langchain_text_splitters.base import split_text_on_tokens
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_text_splitters.html import (
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    HTMLSemanticPreservingSplitter,
)
from langchain_text_splitters.json import RecursiveJsonSplitter
from langchain_text_splitters.jsx import JSFrameworkTextSplitter
from langchain_text_splitters.markdown import (
    ExperimentalMarkdownSyntaxTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_text_splitters.python import PythonCodeTextSplitter

if TYPE_CHECKING:
    from collections.abc import Callable

    from bs4 import Tag

FAKE_PYTHON_TEXT = """
class Foo:

    def bar():


def foo():

def testing_func():

def bar():
"""


def test_character_text_splitter() -> None:
    """Test splitting by character count."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=7, chunk_overlap=3)
    output = splitter.split_text(text)
    expected_output = ["foo bar", "bar baz", "baz 123"]
    assert output == expected_output


def test_character_text_splitter_empty_doc() -> None:
    """Test splitting by character count doesn't create empty documents."""
    text = "foo  bar"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar"]
    assert output == expected_output


def test_character_text_splitter_separtor_empty_doc() -> None:
    """Test edge cases are separators."""
    text = "f b"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = ["f", "b"]
    assert output == expected_output


def test_character_text_splitter_long() -> None:
    """Test splitting by character count on long words."""
    text = "foo bar baz a a"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "a a"]
    assert output == expected_output


def test_character_text_splitter_short_words_first() -> None:
    """Test splitting by character count when shorter words are first."""
    text = "a a foo bar baz"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["a a", "foo", "bar", "baz"]
    assert output == expected_output


def test_character_text_splitter_longer_words() -> None:
    """Test splitting by characters when splits not found easily."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "123"]
    assert output == expected_output


# edge cases
def test_character_text_splitter_no_separator_in_text() -> None:
    """Text splitting where there is no seperator but a single word."""
    text = "singleword"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=10, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = ["singleword"]
    assert output == expected_output


def test_character_text_splitter_no_loop_in_text() -> None:
    """Text splitting safe guards when single word is looped by word length."""
    text = "hello"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=5)
    output = splitter.split_text(text)
    expected_output = ["hello"]
    assert output == expected_output


def test_character_text_splitter_empty_input() -> None:
    """Test splitting safely where there is no input to process."""
    text = ""
    splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = []
    assert output == expected_output


def test_character_text_splitter_whitespace_only() -> None:
    """Test splitting safely where there is white space."""
    text = " "
    splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = []
    assert output == expected_output


@pytest.mark.parametrize(
    ("separator", "is_separator_regex"), [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_keep_separator_regex(
    *, separator: str, is_separator_regex: bool
) -> None:
    """Test CharacterTextSplitter keep separator regex.

    Test splitting by characters while keeping the separator
    that is a regex special character.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator=True,
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo", ".bar", ".baz", ".123"]
    assert output == expected_output


@pytest.mark.parametrize(
    ("separator", "is_separator_regex"), [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_keep_separator_regex_start(
    *, separator: str, is_separator_regex: bool
) -> None:
    """Test CharacterTextSplitter keep separator regex and put at start.

    Test splitting by characters while keeping the separator
    that is a regex special character and placing it at the start of each chunk.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator="start",
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo", ".bar", ".baz", ".123"]
    assert output == expected_output


@pytest.mark.parametrize(
    ("separator", "is_separator_regex"), [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_keep_separator_regex_end(
    *, separator: str, is_separator_regex: bool
) -> None:
    """Test CharacterTextSplitter keep separator regex and put at end.

    Test splitting by characters while keeping the separator
    that is a regex special character and placing it at the end of each chunk.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator="end",
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo.", "bar.", "baz.", "123"]
    assert output == expected_output


@pytest.mark.parametrize(
    ("separator", "is_separator_regex"), [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_discard_separator_regex(
    *, separator: str, is_separator_regex: bool
) -> None:
    """Test CharacterTextSplitter discard separator regex.

    Test splitting by characters discarding the separator
    that is a regex special character.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator=False,
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "123"]
    assert output == expected_output


def test_recursive_character_text_splitter_keep_separators() -> None:
    split_tags = [",", "."]
    query = "Apple,banana,orange and tomato."
    # start
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
        separators=split_tags,
        keep_separator="start",
    )
    result = splitter.split_text(query)
    assert result == ["Apple", ",banana", ",orange and tomato", "."]

    # end
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
        separators=split_tags,
        keep_separator="end",
    )
    result = splitter.split_text(query)
    assert result == ["Apple,", "banana,", "orange and tomato."]


def test_character_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Got a larger chunk overlap (4) than chunk size (2), should be smaller."
        ),
    ):
        CharacterTextSplitter(chunk_size=2, chunk_overlap=4)
    for invalid_size in (0, -1):
        with pytest.raises(ValueError, match="chunk_size must be > 0, got"):
            CharacterTextSplitter(chunk_size=invalid_size)
    with pytest.raises(ValueError, match="chunk_overlap must be >= 0, got -1"):
        CharacterTextSplitter(chunk_size=2, chunk_overlap=-1)


def test_merge_splits() -> None:
    """Test merging splits with a given separator."""
    splitter = CharacterTextSplitter(separator=" ", chunk_size=9, chunk_overlap=2)
    splits = ["foo", "bar", "baz"]
    expected_output = ["foo bar", "baz"]
    output = splitter._merge_splits(splits, separator=" ")
    assert output == expected_output


def test_create_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts)
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


def test_create_documents_with_metadata() -> None:
    """Test create documents with metadata method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts, [{"source": "1"}, {"source": "2"}])
    expected_docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "1"}),
        Document(page_content="baz", metadata={"source": "2"}),
    ]
    assert docs == expected_docs


@pytest.mark.parametrize(
    ("splitter", "text", "expected_docs"),
    [
        (
            CharacterTextSplitter(
                separator=" ", chunk_size=7, chunk_overlap=3, add_start_index=True
            ),
            "foo bar baz 123",
            [
                Document(page_content="foo bar", metadata={"start_index": 0}),
                Document(page_content="bar baz", metadata={"start_index": 4}),
                Document(page_content="baz 123", metadata={"start_index": 8}),
            ],
        ),
        (
            RecursiveCharacterTextSplitter(
                chunk_size=6,
                chunk_overlap=0,
                separators=["\n\n", "\n", " ", ""],
                add_start_index=True,
            ),
            "w1 w1 w1 w1 w1 w1 w1 w1 w1",
            [
                Document(page_content="w1 w1", metadata={"start_index": 0}),
                Document(page_content="w1 w1", metadata={"start_index": 6}),
                Document(page_content="w1 w1", metadata={"start_index": 12}),
                Document(page_content="w1 w1", metadata={"start_index": 18}),
                Document(page_content="w1", metadata={"start_index": 24}),
            ],
        ),
    ],
)
def test_create_documents_with_start_index(
    splitter: TextSplitter, text: str, expected_docs: list[Document]
) -> None:
    """Test create documents method."""
    docs = splitter.create_documents([text])
    assert docs == expected_docs
    for doc in docs:
        s_i = doc.metadata["start_index"]
        assert text[s_i : s_i + len(doc.page_content)] == doc.page_content


def test_metadata_not_shallow() -> None:
    """Test that metadatas are not shallow."""
    texts = ["foo bar"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts, [{"source": "1"}])
    expected_docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "1"}),
    ]
    assert docs == expected_docs
    docs[0].metadata["foo"] = 1
    assert docs[0].metadata == {"source": "1", "foo": 1}
    assert docs[1].metadata == {"source": "1"}


def test_iterative_text_splitter_keep_separator() -> None:
    chunk_size = 5
    output = __test_iterative_text_splitter(chunk_size=chunk_size, keep_separator=True)

    assert output == [
        "....5",
        "X..3",
        "Y...4",
        "X....5",
        "Y...",
    ]


def test_iterative_text_splitter_discard_separator() -> None:
    chunk_size = 5
    output = __test_iterative_text_splitter(chunk_size=chunk_size, keep_separator=False)

    assert output == [
        "....5",
        "..3",
        "...4",
        "....5",
        "...",
    ]


def __test_iterative_text_splitter(
    *, chunk_size: int, keep_separator: bool
) -> list[str]:
    chunk_size += 1 if keep_separator else 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["X", "Y"],
        keep_separator=keep_separator,
    )
    text = "....5X..3Y...4X....5Y..."
    output = splitter.split_text(text)
    for chunk in output:
        assert len(chunk) <= chunk_size, f"Chunk is larger than {chunk_size}"
    return output


def test_iterative_text_splitter() -> None:
    """Test iterative text splitter."""
    text = """Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.

Bye!\n\n-H."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = [
        "Hi.",
        "I'm",
        "Harrison.",
        "How? Are?",
        "You?",
        "Okay then",
        "f f f f.",
        "This is a",
        "weird",
        "text to",
        "write,",
        "but gotta",
        "test the",
        "splitting",
        "gggg",
        "some how.",
        "Bye!",
        "-H.",
    ]
    assert output == expected_output


def test_split_documents() -> None:
    """Test split_documents."""
    splitter = CharacterTextSplitter(separator="", chunk_size=1, chunk_overlap=0)
    docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "2"}),
        Document(page_content="baz", metadata={"source": "1"}),
    ]
    expected_output = [
        Document(page_content="f", metadata={"source": "1"}),
        Document(page_content="o", metadata={"source": "1"}),
        Document(page_content="o", metadata={"source": "1"}),
        Document(page_content="b", metadata={"source": "2"}),
        Document(page_content="a", metadata={"source": "2"}),
        Document(page_content="r", metadata={"source": "2"}),
        Document(page_content="b", metadata={"source": "1"}),
        Document(page_content="a", metadata={"source": "1"}),
        Document(page_content="z", metadata={"source": "1"}),
    ]
    assert splitter.split_documents(docs) == expected_output


def test_python_text_splitter() -> None:
    splitter = PythonCodeTextSplitter(chunk_size=30, chunk_overlap=0)
    splits = splitter.split_text(FAKE_PYTHON_TEXT)
    split_0 = """class Foo:\n\n    def bar():"""
    split_1 = """def foo():"""
    split_2 = """def testing_func():"""
    split_3 = """def bar():"""
    expected_splits = [split_0, split_1, split_2, split_3]
    assert splits == expected_splits


FAKE_JSX_TEXT = """
import React from 'react';
import OtherComponent from './OtherComponent';

function MyComponent() {
  const [count, setCount] = React.useState(0);

  const handleClick = () => {
    setCount(count + 1);
  };

  return (
    <div>
      <h1>Counter: {count}</h1>
      <button onClick={handleClick}>
        Increment
      </button>
      <OtherComponent />
    </div>
  );
}

export default MyComponent;
"""


def test_jsx_text_splitter() -> None:
    splitter = JSFrameworkTextSplitter(chunk_size=30, chunk_overlap=0)
    splits = splitter.split_text(FAKE_JSX_TEXT)

    expected_splits = [
        "\nimport React from 'react';\n"
        "import OtherComponent from './OtherComponent';\n",
        "\nfunction MyComponent() {\n  const [count, setCount] = React.useState(0);",
        "\n\n  const handleClick = () => {\n    setCount(count + 1);\n  };",
        "return (",
        "<div>",
        "<h1>Counter: {count}</h1>\n      ",
        "<button onClick={handleClick}>\n        Increment\n      </button>\n      ",
        "<OtherComponent />\n    </div>\n  );\n}\n",
        "export default MyComponent;",
    ]
    assert [s.strip() for s in splits] == [s.strip() for s in expected_splits]


FAKE_VUE_TEXT = """
<template>
  <div>
    <h1>{{ title }}</h1>
    <button @click="increment">
      Count is: {{ count }}
    </button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      title: 'Counter App',
      count: 0
    }
  },
  methods: {
    increment() {
      this.count++
    }
  }
}
</script>

<style>
button {
  color: blue;
}
</style>
"""


def test_vue_text_splitter() -> None:
    splitter = JSFrameworkTextSplitter(chunk_size=30, chunk_overlap=0)
    splits = splitter.split_text(FAKE_VUE_TEXT)

    expected_splits = [
        "<template>",
        "<div>",
        "<h1>{{ title }}</h1>",
        '<button @click="increment">\n      Count is: {{ count }}\n'
        "    </button>\n  </div>\n</template>",
        "<script>",
        "export",
        " default {\n  data() {\n    return {\n      title: 'Counter App',\n      "
        "count: 0\n    }\n  },\n  methods: {\n    increment() {\n      "
        "this.count++\n    }\n  }\n}\n</script>",
        "<style>\nbutton {\n  color: blue;\n}\n</style>",
    ]
    assert [s.strip() for s in splits] == [s.strip() for s in expected_splits]


FAKE_SVELTE_TEXT = """
<script>
  let count = 0

  function increment() {
    count += 1
  }
</script>

<main>
  <h1>Counter App</h1>
  <button on:click={increment}>
    Count is: {count}
  </button>
</main>

<style>
  button {
    color: blue;
  }
</style>
"""


def test_svelte_text_splitter() -> None:
    splitter = JSFrameworkTextSplitter(chunk_size=30, chunk_overlap=0)
    splits = splitter.split_text(FAKE_SVELTE_TEXT)

    expected_splits = [
        "<script>\n  let count = 0",
        "\n\n  function increment() {\n    count += 1\n  }\n</script>",
        "<main>",
        "<h1>Counter App</h1>",
        "<button on:click={increment}>\n    Count is: {count}\n  </button>\n</main>",
        "<style>\n  button {\n    color: blue;\n  }\n</style>",
    ]
    assert [s.strip() for s in splits] == [s.strip() for s in expected_splits]


CHUNK_SIZE = 16


def test_python_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "def",
        "hello_world():",
        'print("Hello,',
        'World!")',
        "# Call the",
        "function",
        "hello_world()",
    ]


def test_golang_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.GO, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
package main

import "fmt"

func helloWorld() {
    fmt.Println("Hello, World!")
}

func main() {
    helloWorld()
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "package main",
        'import "fmt"',
        "func",
        "helloWorld() {",
        'fmt.Println("He',
        "llo,",
        'World!")',
        "}",
        "func main() {",
        "helloWorld()",
        "}",
    ]


def test_rst_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.RST, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
Sample Document
===============

Section
-------

This is the content of the section.

Lists
-----

- Item 1
- Item 2
- Item 3

Comment
*******
Not a comment

.. This is a comment
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "Sample Document",
        "===============",
        "Section",
        "-------",
        "This is the",
        "content of the",
        "section.",
        "Lists",
        "-----",
        "- Item 1",
        "- Item 2",
        "- Item 3",
        "Comment",
        "*******",
        "Not a comment",
        ".. This is a",
        "comment",
    ]
    # Special test for special characters
    code = "harry\n***\nbabylon is"
    chunks = splitter.split_text(code)
    assert chunks == ["harry", "***\nbabylon is"]


def test_proto_file_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PROTO, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
syntax = "proto3";

package example;

message Person {
    string name = 1;
    int32 age = 2;
    repeated string hobbies = 3;
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "syntax =",
        '"proto3";',
        "package",
        "example;",
        "message Person",
        "{",
        "string name",
        "= 1;",
        "int32 age =",
        "2;",
        "repeated",
        "string hobbies",
        "= 3;",
        "}",
    ]


def test_javascript_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.JS, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
function helloWorld() {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "function",
        "helloWorld() {",
        'console.log("He',
        "llo,",
        'World!");',
        "}",
        "// Call the",
        "function",
        "helloWorld();",
    ]


def test_cobol_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.COBOL, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
IDENTIFICATION DIVISION.
PROGRAM-ID. HelloWorld.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 GREETING           PIC X(12)   VALUE 'Hello, World!'.
PROCEDURE DIVISION.
DISPLAY GREETING.
STOP RUN.
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "IDENTIFICATION",
        "DIVISION.",
        "PROGRAM-ID.",
        "HelloWorld.",
        "DATA DIVISION.",
        "WORKING-STORAGE",
        "SECTION.",
        "01 GREETING",
        "PIC X(12)",
        "VALUE 'Hello,",
        "World!'.",
        "PROCEDURE",
        "DIVISION.",
        "DISPLAY",
        "GREETING.",
        "STOP RUN.",
    ]


def test_typescript_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.TS, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
function helloWorld(): void {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "function",
        "helloWorld():",
        "void {",
        'console.log("He',
        "llo,",
        'World!");',
        "}",
        "// Call the",
        "function",
        "helloWorld();",
    ]


def test_java_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.JAVA, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "public class",
        "HelloWorld {",
        "public",
        "static void",
        "main(String[]",
        "args) {",
        "System.out.prin",
        'tln("Hello,',
        'World!");',
        "}\n}",
    ]


def test_kotlin_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.KOTLIN, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
class HelloWorld {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            println("Hello, World!")
        }
    }
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "class",
        "HelloWorld {",
        "companion",
        "object {",
        "@JvmStatic",
        "fun",
        "main(args:",
        "Array<String>)",
        "{",
        'println("Hello,',
        'World!")',
        "}\n    }",
        "}",
    ]


def test_csharp_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.CSHARP, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
using System;
class Program
{
    static void Main()
    {
        int age = 30; // Change the age value as needed

        // Categorize the age without any console output
        if (age < 18)
        {
            // Age is under 18
        }
        else if (age >= 18 && age < 65)
        {
            // Age is an adult
        }
        else
        {
            // Age is a senior citizen
        }
    }
}
    """

    chunks = splitter.split_text(code)
    assert chunks == [
        "using System;",
        "class Program\n{",
        "static void",
        "Main()",
        "{",
        "int age",
        "= 30; // Change",
        "the age value",
        "as needed",
        "//",
        "Categorize the",
        "age without any",
        "console output",
        "if (age",
        "< 18)",
        "{",
        "//",
        "Age is under 18",
        "}",
        "else if",
        "(age >= 18 &&",
        "age < 65)",
        "{",
        "//",
        "Age is an adult",
        "}",
        "else",
        "{",
        "//",
        "Age is a senior",
        "citizen",
        "}\n    }",
        "}",
    ]


def test_cpp_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.CPP, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "#include",
        "<iostream>",
        "int main() {",
        "std::cout",
        '<< "Hello,',
        'World!" <<',
        "std::endl;",
        "return 0;\n}",
    ]


def test_scala_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.SCALA, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
object HelloWorld {
  def main(args: Array[String]): Unit = {
    println("Hello, World!")
  }
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "object",
        "HelloWorld {",
        "def",
        "main(args:",
        "Array[String]):",
        "Unit = {",
        'println("Hello,',
        'World!")',
        "}\n}",
    ]


def test_ruby_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.RUBY, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
def hello_world
  puts "Hello, World!"
end

hello_world
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "def hello_world",
        'puts "Hello,',
        'World!"',
        "end",
        "hello_world",
    ]


def test_php_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PHP, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
<?php
function hello_world() {
    echo "Hello, World!";
}

hello_world();
?>
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "<?php",
        "function",
        "hello_world() {",
        "echo",
        '"Hello,',
        'World!";',
        "}",
        "hello_world();",
        "?>",
    ]


def test_swift_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.SWIFT, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
func helloWorld() {
    print("Hello, World!")
}

helloWorld()
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "func",
        "helloWorld() {",
        'print("Hello,',
        'World!")',
        "}",
        "helloWorld()",
    ]


def test_rust_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.RUST, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
fn main() {
    println!("Hello, World!");
}
    """
    chunks = splitter.split_text(code)
    assert chunks == ["fn main() {", 'println!("Hello', ",", 'World!");', "}"]


def test_r_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.R, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
library(dplyr)

my_func <- function(x) {
    return(x + 1)
}

if (TRUE) {
    print("Hello")
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "library(dplyr)",
        "my_func <-",
        "function(x) {",
        "return(x +",
        "1)",
        "}",
        "if (TRUE) {",
        'print("Hello")',
        "}",
    ]


def test_markdown_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
# Sample Document

## Section

This is the content of the section.

## Lists

- Item 1
- Item 2
- Item 3

### Horizontal lines

***********
____________
-------------------

#### Code blocks
```
This is a code block

# sample code
a = 1
b = 2
```
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "# Sample",
        "Document",
        "## Section",
        "This is the",
        "content of the",
        "section.",
        "## Lists",
        "- Item 1",
        "- Item 2",
        "- Item 3",
        "### Horizontal",
        "lines",
        "***********",
        "____________",
        "---------------",
        "----",
        "#### Code",
        "blocks",
        "```",
        "This is a code",
        "block",
        "# sample code",
        "a = 1\nb = 2",
        "```",
    ]
    # Special test for special characters
    code = "harry\n***\nbabylon is"
    chunks = splitter.split_text(code)
    assert chunks == ["harry", "***\nbabylon is"]


def test_latex_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.LATEX, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
Hi Harrison!
\\chapter{1}
"""
    chunks = splitter.split_text(code)
    assert chunks == ["Hi Harrison!", "\\chapter{1}"]


def test_html_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.HTML, chunk_size=60, chunk_overlap=0
    )
    code = """
<h1>Sample Document</h1>
    <h2>Section</h2>
        <p id="1234">Reference content.</p>

    <h2>Lists</h2>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>

        <h3>A block</h3>
            <div class="amazing">
                <p>Some text</p>
                <p>Some more text</p>
            </div>
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "<h1>Sample Document</h1>\n    <h2>Section</h2>",
        '<p id="1234">Reference content.</p>',
        "<h2>Lists</h2>\n        <ul>",
        "<li>Item 1</li>\n            <li>Item 2</li>",
        "<li>Item 3</li>\n        </ul>",
        "<h3>A block</h3>",
        '<div class="amazing">',
        "<p>Some text</p>",
        "<p>Some more text</p>\n            </div>",
    ]


def test_md_header_text_splitter_1() -> None:
    """Test markdown splitter by header: Case 1."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bar\n\n"
        "Hi this is Jim\n\n"
        "Hi this is Joe\n\n"
        " ## Baz\n\n"
        " Hi this is Molly"
    )
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)
    expected_output = [
        Document(
            page_content="Hi this is Jim  \nHi this is Joe",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
        Document(
            page_content="Hi this is Molly",
            metadata={"Header 1": "Foo", "Header 2": "Baz"},
        ),
    ]
    assert output == expected_output


def test_md_header_text_splitter_2() -> None:
    """Test markdown splitter by header: Case 2."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bar\n\n"
        "Hi this is Jim\n\n"
        "Hi this is Joe\n\n"
        " ### Boo \n\n"
        " Hi this is Lance \n\n"
        " ## Baz\n\n"
        " Hi this is Molly"
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)
    expected_output = [
        Document(
            page_content="Hi this is Jim  \nHi this is Joe",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
        Document(
            page_content="Hi this is Lance",
            metadata={"Header 1": "Foo", "Header 2": "Bar", "Header 3": "Boo"},
        ),
        Document(
            page_content="Hi this is Molly",
            metadata={"Header 1": "Foo", "Header 2": "Baz"},
        ),
    ]
    assert output == expected_output


def test_md_header_text_splitter_3() -> None:
    """Test markdown splitter by header: Case 3."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bar\n\n"
        "Hi this is Jim\n\n"
        "Hi this is Joe\n\n"
        " ### Boo \n\n"
        " Hi this is Lance \n\n"
        " #### Bim \n\n"
        " Hi this is John \n\n"
        " ## Baz\n\n"
        " Hi this is Molly"
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content="Hi this is Jim  \nHi this is Joe",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
        Document(
            page_content="Hi this is Lance",
            metadata={"Header 1": "Foo", "Header 2": "Bar", "Header 3": "Boo"},
        ),
        Document(
            page_content="Hi this is John",
            metadata={
                "Header 1": "Foo",
                "Header 2": "Bar",
                "Header 3": "Boo",
                "Header 4": "Bim",
            },
        ),
        Document(
            page_content="Hi this is Molly",
            metadata={"Header 1": "Foo", "Header 2": "Baz"},
        ),
    ]

    assert output == expected_output


def test_md_header_text_splitter_preserve_headers_1() -> None:
    """Test markdown splitter by header: Preserve Headers."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bat\n\n"
        "Hi this is Jim\n\n"
        "Hi Joe\n\n"
        "## Baz\n\n"
        "# Bar\n\n"
        "This is Alice\n\n"
        "This is Bob"
    )
    headers_to_split_on = [
        ("#", "Header 1"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    output = markdown_splitter.split_text(markdown_document)
    expected_output = [
        Document(
            page_content="# Foo  \n## Bat  \nHi this is Jim  \nHi Joe  \n## Baz",
            metadata={"Header 1": "Foo"},
        ),
        Document(
            page_content="# Bar  \nThis is Alice  \nThis is Bob",
            metadata={"Header 1": "Bar"},
        ),
    ]
    assert output == expected_output


def test_md_header_text_splitter_preserve_headers_2() -> None:
    """Test markdown splitter by header: Preserve Headers."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bar\n\n"
        "Hi this is Jim\n\n"
        "Hi this is Joe\n\n"
        "### Boo \n\n"
        "Hi this is Lance\n\n"
        "## Baz\n\n"
        "Hi this is Molly\n"
        "    ## Buz\n"
        "# Bop"
    )
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    output = markdown_splitter.split_text(markdown_document)
    expected_output = [
        Document(
            page_content="# Foo  \n## Bar  \nHi this is Jim  \nHi this is Joe",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
        Document(
            page_content="### Boo  \nHi this is Lance",
            metadata={"Header 1": "Foo", "Header 2": "Bar", "Header 3": "Boo"},
        ),
        Document(
            page_content="## Baz  \nHi this is Molly",
            metadata={"Header 1": "Foo", "Header 2": "Baz"},
        ),
        Document(
            page_content="## Buz",
            metadata={"Header 1": "Foo", "Header 2": "Buz"},
        ),
        Document(page_content="# Bop", metadata={"Header 1": "Bop"}),
    ]
    assert output == expected_output


@pytest.mark.parametrize("fence", [("```"), ("~~~")])
def test_md_header_text_splitter_fenced_code_block(fence: str) -> None:
    """Test markdown splitter by header: Fenced code block."""
    markdown_document = (
        f"# This is a Header\n\n{fence}\nfoo()\n# Not a header\nbar()\n{fence}"
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content=f"{fence}\nfoo()\n# Not a header\nbar()\n{fence}",
            metadata={"Header 1": "This is a Header"},
        ),
    ]

    assert output == expected_output


@pytest.mark.parametrize(("fence", "other_fence"), [("```", "~~~"), ("~~~", "```")])
def test_md_header_text_splitter_fenced_code_block_interleaved(
    fence: str, other_fence: str
) -> None:
    """Test markdown splitter by header: Interleaved fenced code block."""
    markdown_document = (
        "# This is a Header\n\n"
        f"{fence}\n"
        "foo\n"
        "# Not a header\n"
        f"{other_fence}\n"
        "# Not a header\n"
        f"{fence}"
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content=(
                f"{fence}\nfoo\n# Not a header\n{other_fence}\n# Not a header\n{fence}"
            ),
            metadata={"Header 1": "This is a Header"},
        ),
    ]

    assert output == expected_output


@pytest.mark.parametrize("characters", ["\ufeff"])
def test_md_header_text_splitter_with_invisible_characters(characters: str) -> None:
    """Test markdown splitter by header: Fenced code block."""
    markdown_document = f"{characters}# Foo\n\nfoo()\n{characters}## Bar\n\nbar()"

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content="foo()",
            metadata={"Header 1": "Foo"},
        ),
        Document(
            page_content="bar()",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
    ]

    assert output == expected_output


def test_md_header_text_splitter_with_custom_headers() -> None:
    """Test markdown splitter with custom header patterns like **Header**."""
    markdown_document = """**Chapter 1**

This is the content for chapter 1.

***Section 1.1***

This is the content for section 1.1.

**Chapter 2**

This is the content for chapter 2.

***Section 2.1***

This is the content for section 2.1.
"""

    headers_to_split_on = [
        ("**", "Bold Header"),
        ("***", "Bold Italic Header"),
    ]

    custom_header_patterns = {
        "**": 1,  # Level 1 headers
        "***": 2,  # Level 2 headers
    }
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        custom_header_patterns=custom_header_patterns,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content="This is the content for chapter 1.",
            metadata={"Bold Header": "Chapter 1"},
        ),
        Document(
            page_content="This is the content for section 1.1.",
            metadata={"Bold Header": "Chapter 1", "Bold Italic Header": "Section 1.1"},
        ),
        Document(
            page_content="This is the content for chapter 2.",
            metadata={"Bold Header": "Chapter 2"},
        ),
        Document(
            page_content="This is the content for section 2.1.",
            metadata={"Bold Header": "Chapter 2", "Bold Italic Header": "Section 2.1"},
        ),
    ]

    assert output == expected_output


def test_md_header_text_splitter_mixed_headers() -> None:
    """Test markdown splitter with both standard and custom headers."""
    markdown_document = """# Standard Header 1

Content under standard header.

**Custom Header 1**

Content under custom header.

## Standard Header 2

Content under standard header 2.

***Custom Header 2***

Content under custom header 2.
"""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("**", "Bold Header"),
        ("***", "Bold Italic Header"),
    ]

    custom_header_patterns = {
        "**": 1,  # Same level as #
        "***": 2,  # Same level as ##
    }

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        custom_header_patterns=custom_header_patterns,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content="Content under standard header.",
            metadata={"Header 1": "Standard Header 1"},
        ),
        Document(
            page_content="Content under custom header.",
            metadata={"Bold Header": "Custom Header 1"},
        ),
        Document(
            page_content="Content under standard header 2.",
            metadata={
                "Bold Header": "Custom Header 1",
                "Header 2": "Standard Header 2",
            },
        ),
        Document(
            page_content="Content under custom header 2.",
            metadata={
                "Bold Header": "Custom Header 1",
                "Bold Italic Header": "Custom Header 2",
            },
        ),
    ]

    assert output == expected_output


EXPERIMENTAL_MARKDOWN_DOCUMENT = (
    "# My Header 1\n"
    "Content for header 1\n"
    "## Header 2\n"
    "Content for header 2\n"
    "### Header 3\n"
    "Content for header 3\n"
    "## Header 2 Again\n"
    "This should be tagged with Header 1 and Header 2 Again\n"
    "```python\n"
    "def func_definition():\n"
    "   print('Keep the whitespace consistent')\n"
    "```\n"
    "# Header 1 again\n"
    "We should also split on the horizontal line\n"
    "----\n"
    "This will be a new doc but with the same header metadata\n\n"
    "And it includes a new paragraph"
)


def test_experimental_markdown_syntax_text_splitter() -> None:
    """Test experimental markdown syntax splitter."""
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter()
    output = markdown_splitter.split_text(EXPERIMENTAL_MARKDOWN_DOCUMENT)

    expected_output = [
        Document(
            page_content="Content for header 1\n",
            metadata={"Header 1": "My Header 1"},
        ),
        Document(
            page_content="Content for header 2\n",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2"},
        ),
        Document(
            page_content="Content for header 3\n",
            metadata={
                "Header 1": "My Header 1",
                "Header 2": "Header 2",
                "Header 3": "Header 3",
            },
        ),
        Document(
            page_content="This should be tagged with Header 1 and Header 2 Again\n",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2 Again"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_header_configuration() -> None:
    """Test experimental markdown syntax splitter."""
    headers_to_split_on = [("#", "Encabezamiento 1")]

    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    output = markdown_splitter.split_text(EXPERIMENTAL_MARKDOWN_DOCUMENT)

    expected_output = [
        Document(
            page_content=(
                "Content for header 1\n"
                "## Header 2\n"
                "Content for header 2\n"
                "### Header 3\n"
                "Content for header 3\n"
                "## Header 2 Again\n"
                "This should be tagged with Header 1 and Header 2 Again\n"
            ),
            metadata={"Encabezamiento 1": "My Header 1"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={"Code": "python", "Encabezamiento 1": "My Header 1"},
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Encabezamiento 1": "Header 1 again"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Encabezamiento 1": "Header 1 again"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_with_headers() -> None:
    """Test experimental markdown syntax splitter."""
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(strip_headers=False)
    output = markdown_splitter.split_text(EXPERIMENTAL_MARKDOWN_DOCUMENT)

    expected_output = [
        Document(
            page_content="# My Header 1\nContent for header 1\n",
            metadata={"Header 1": "My Header 1"},
        ),
        Document(
            page_content="## Header 2\nContent for header 2\n",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2"},
        ),
        Document(
            page_content="### Header 3\nContent for header 3\n",
            metadata={
                "Header 1": "My Header 1",
                "Header 2": "Header 2",
                "Header 3": "Header 3",
            },
        ),
        Document(
            page_content=(
                "## Header 2 Again\n"
                "This should be tagged with Header 1 and Header 2 Again\n"
            ),
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2 Again"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content=(
                "# Header 1 again\nWe should also split on the horizontal line\n"
            ),
            metadata={"Header 1": "Header 1 again"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_split_lines() -> None:
    """Test experimental markdown syntax splitter."""
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(return_each_line=True)
    output = markdown_splitter.split_text(EXPERIMENTAL_MARKDOWN_DOCUMENT)

    expected_output = [
        Document(
            page_content="Content for header 1", metadata={"Header 1": "My Header 1"}
        ),
        Document(
            page_content="Content for header 2",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2"},
        ),
        Document(
            page_content="Content for header 3",
            metadata={
                "Header 1": "My Header 1",
                "Header 2": "Header 2",
                "Header 3": "Header 3",
            },
        ),
        Document(
            page_content="This should be tagged with Header 1 and Header 2 Again",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2 Again"},
        ),
        Document(
            page_content="```python",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="def func_definition():",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="   print('Keep the whitespace consistent')",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="```",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line",
            metadata={"Header 1": "Header 1 again"},
        ),
        Document(
            page_content="This will be a new doc but with the same header metadata",
            metadata={"Header 1": "Header 1 again"},
        ),
        Document(
            page_content="And it includes a new paragraph",
            metadata={"Header 1": "Header 1 again"},
        ),
    ]

    assert output == expected_output


EXPERIMENTAL_MARKDOWN_DOCUMENTS = [
    (
        "# My Header 1 From Document 1\n"
        "Content for header 1 from Document 1\n"
        "## Header 2 From Document 1\n"
        "Content for header 2 from Document 1\n"
        "```python\n"
        "def func_definition():\n"
        "   print('Keep the whitespace consistent')\n"
        "```\n"
        "# Header 1 again From Document 1\n"
        "We should also split on the horizontal line\n"
        "----\n"
        "This will be a new doc but with the same header metadata\n\n"
        "And it includes a new paragraph"
    ),
    (
        "# My Header 1 From Document 2\n"
        "Content for header 1 from Document 2\n"
        "## Header 2 From Document 2\n"
        "Content for header 2 from Document 2\n"
        "```python\n"
        "def func_definition():\n"
        "   print('Keep the whitespace consistent')\n"
        "```\n"
        "# Header 1 again From Document 2\n"
        "We should also split on the horizontal line\n"
        "----\n"
        "This will be a new doc but with the same header metadata\n\n"
        "And it includes a new paragraph"
    ),
]


def test_experimental_markdown_syntax_text_splitter_on_multi_files() -> None:
    """Test ExperimentalMarkdownSyntaxTextSplitter on multiple files.

    Test experimental markdown syntax splitter split on default called consecutively
    on two files.
    """
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter()
    output = []
    for experimental_markdown_document in EXPERIMENTAL_MARKDOWN_DOCUMENTS:
        output += markdown_splitter.split_text(experimental_markdown_document)

    expected_output = [
        Document(
            page_content="Content for header 1 from Document 1\n",
            metadata={"Header 1": "My Header 1 From Document 1"},
        ),
        Document(
            page_content="Content for header 2 from Document 1\n",
            metadata={
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="Content for header 1 from Document 2\n",
            metadata={"Header 1": "My Header 1 From Document 2"},
        ),
        Document(
            page_content="Content for header 2 from Document 2\n",
            metadata={
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_split_lines_on_multi_files() -> (
    None
):
    """Test ExperimentalMarkdownSyntaxTextSplitter split lines on multiple files.

    Test experimental markdown syntax splitter split on each line called consecutively
    on two files.
    """
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(return_each_line=True)
    output = []
    for experimental_markdown_document in EXPERIMENTAL_MARKDOWN_DOCUMENTS:
        output += markdown_splitter.split_text(experimental_markdown_document)
    expected_output = [
        Document(
            page_content="Content for header 1 from Document 1",
            metadata={"Header 1": "My Header 1 From Document 1"},
        ),
        Document(
            page_content="Content for header 2 from Document 1",
            metadata={
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="```python",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="def func_definition():",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="   print('Keep the whitespace consistent')",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="```",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="This will be a new doc but with the same header metadata",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="And it includes a new paragraph",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="Content for header 1 from Document 2",
            metadata={"Header 1": "My Header 1 From Document 2"},
        ),
        Document(
            page_content="Content for header 2 from Document 2",
            metadata={
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="```python",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="def func_definition():",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="   print('Keep the whitespace consistent')",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="```",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content="This will be a new doc but with the same header metadata",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content="And it includes a new paragraph",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_with_header_on_multi_files() -> (
    None
):
    """Test ExperimentalMarkdownSyntaxTextSplitter with header on multiple files.

    Test experimental markdown splitter by header called consecutively on two files.
    """
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(strip_headers=False)
    output = []
    for experimental_markdown_document in EXPERIMENTAL_MARKDOWN_DOCUMENTS:
        output += markdown_splitter.split_text(experimental_markdown_document)

    expected_output = [
        Document(
            page_content="# My Header 1 From Document 1\n"
            "Content for header 1 from Document 1\n",
            metadata={"Header 1": "My Header 1 From Document 1"},
        ),
        Document(
            page_content="## Header 2 From Document 1\n"
            "Content for header 2 from Document 1\n",
            metadata={
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="# Header 1 again From Document 1\n"
            "We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="# My Header 1 From Document 2\n"
            "Content for header 1 from Document 2\n",
            metadata={"Header 1": "My Header 1 From Document 2"},
        ),
        Document(
            page_content="## Header 2 From Document 2\n"
            "Content for header 2 from Document 2\n",
            metadata={
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="# Header 1 again From Document 2\n"
            "We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
    ]
    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_header_config_on_multi_files() -> (
    None
):
    """Test ExperimentalMarkdownSyntaxTextSplitter header config on multiple files.

    Test experimental markdown splitter by header configuration called consecutively
    on two files.
    """
    headers_to_split_on = [("#", "Encabezamiento 1")]
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    output = []
    for experimental_markdown_document in EXPERIMENTAL_MARKDOWN_DOCUMENTS:
        output += markdown_splitter.split_text(experimental_markdown_document)

    expected_output = [
        Document(
            page_content="Content for header 1 from Document 1\n"
            "## Header 2 From Document 1\n"
            "Content for header 2 from Document 1\n",
            metadata={"Encabezamiento 1": "My Header 1 From Document 1"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Encabezamiento 1": "My Header 1 From Document 1",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Encabezamiento 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Encabezamiento 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="Content for header 1 from Document 2\n"
            "## Header 2 From Document 2\n"
            "Content for header 2 from Document 2\n",
            metadata={"Encabezamiento 1": "My Header 1 From Document 2"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Encabezamiento 1": "My Header 1 From Document 2",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Encabezamiento 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Encabezamiento 1": "Header 1 again From Document 2"},
        ),
    ]

    assert output == expected_output


def test_solidity_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.SOL, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """pragma solidity ^0.8.20;
  contract HelloWorld {
    function add(uint a, uint b) pure public returns(uint) {
      return  a + b;
    }
  }
  """
    chunks = splitter.split_text(code)
    assert chunks == [
        "pragma solidity",
        "^0.8.20;",
        "contract",
        "HelloWorld {",
        "function",
        "add(uint a,",
        "uint b) pure",
        "public",
        "returns(uint) {",
        "return  a",
        "+ b;",
        "}\n  }",
    ]


def test_lua_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.LUA, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
local variable = 10

function add(a, b)
    return a + b
end

if variable > 5 then
    for i=1, variable do
        while i < variable do
            repeat
                print(i)
                i = i + 1
            until i >= variable
        end
    end
end
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "local variable",
        "= 10",
        "function add(a,",
        "b)",
        "return a +",
        "b",
        "end",
        "if variable > 5",
        "then",
        "for i=1,",
        "variable do",
        "while i",
        "< variable do",
        "repeat",
        "print(i)",
        "i = i + 1",
        "until i >=",
        "variable",
        "end",
        "end\nend",
    ]


def test_haskell_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.HASKELL, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
        main :: IO ()
        main = do
          putStrLn "Hello, World!"

        -- Some sample functions
        add :: Int -> Int -> Int
        add x y = x + y
    """
    # Adjusted expected chunks to account for indentation and newlines
    expected_chunks = [
        "main ::",
        "IO ()",
        "main = do",
        "putStrLn",
        '"Hello, World!"',
        "--",
        "Some sample",
        "functions",
        "add :: Int ->",
        "Int -> Int",
        "add x y = x",
        "+ y",
    ]
    chunks = splitter.split_text(code)
    assert chunks == expected_chunks


@pytest.fixture
def html_header_splitter_splitter_factory() -> Callable[
    [list[tuple[str, str]]], HTMLHeaderTextSplitter
]:
    """Fixture to create an `HTMLHeaderTextSplitter` instance with given headers.

    This factory allows dynamic creation of splitters with different headers.
    """

    def _create_splitter(
        headers_to_split_on: list[tuple[str, str]],
    ) -> HTMLHeaderTextSplitter:
        return HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    return _create_splitter


@pytest.mark.parametrize(
    ("headers_to_split_on", "html_input", "expected_documents", "test_case"),
    [
        (
            # Test Case 1: Split on h1 and h2
            [("h1", "Header 1"), ("h2", "Header 2")],
            """
            <html>
                <body>
                    <h1>Introduction</h1>
                    <p>This is the introduction.</p>
                    <h2>Background</h2>
                    <p>Background information.</p>
                    <h1>Conclusion</h1>
                    <p>Final thoughts.</p>
                </body>
            </html>
            """,
            [
                Document(
                    page_content="Introduction", metadata={"Header 1": "Introduction"}
                ),
                Document(
                    page_content="This is the introduction.",
                    metadata={"Header 1": "Introduction"},
                ),
                Document(
                    page_content="Background",
                    metadata={"Header 1": "Introduction", "Header 2": "Background"},
                ),
                Document(
                    page_content="Background information.",
                    metadata={"Header 1": "Introduction", "Header 2": "Background"},
                ),
                Document(
                    page_content="Conclusion", metadata={"Header 1": "Conclusion"}
                ),
                Document(
                    page_content="Final thoughts.", metadata={"Header 1": "Conclusion"}
                ),
            ],
            "Simple headers and paragraphs",
        ),
        (
            # Test Case 2: Nested headers with h1, h2, and h3
            [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
            """
            <html>
                <body>
                    <div>
                        <h1>Main Title</h1>
                        <div>
                            <h2>Subsection</h2>
                            <p>Details of subsection.</p>
                            <div>
                                <h3>Sub-subsection</h3>
                                <p>More details.</p>
                            </div>
                        </div>
                    </div>
                    <h1>Another Main Title</h1>
                    <p>Content under another main title.</p>
                </body>
            </html>
            """,
            [
                Document(
                    page_content="Main Title", metadata={"Header 1": "Main Title"}
                ),
                Document(
                    page_content="Subsection",
                    metadata={"Header 1": "Main Title", "Header 2": "Subsection"},
                ),
                Document(
                    page_content="Details of subsection.",
                    metadata={"Header 1": "Main Title", "Header 2": "Subsection"},
                ),
                Document(
                    page_content="Sub-subsection",
                    metadata={
                        "Header 1": "Main Title",
                        "Header 2": "Subsection",
                        "Header 3": "Sub-subsection",
                    },
                ),
                Document(
                    page_content="More details.",
                    metadata={
                        "Header 1": "Main Title",
                        "Header 2": "Subsection",
                        "Header 3": "Sub-subsection",
                    },
                ),
                Document(
                    page_content="Another Main Title",
                    metadata={"Header 1": "Another Main Title"},
                ),
                Document(
                    page_content="Content under another main title.",
                    metadata={"Header 1": "Another Main Title"},
                ),
            ],
            "Nested headers with h1, h2, and h3",
        ),
        (
            # Test Case 3: No headers
            [("h1", "Header 1")],
            """
            <html>
                <body>
                    <p>Paragraph one.</p>
                    <p>Paragraph two.</p>
                    <div>
                        <p>Paragraph three.</p>
                    </div>
                </body>
            </html>
            """,
            [
                Document(
                    page_content="Paragraph one.  \nParagraph two.  \nParagraph three.",
                    metadata={},
                )
            ],
            "No headers present",
        ),
        (
            # Test Case 4: Multiple headers of the same level
            [("h1", "Header 1")],
            """
            <html>
                <body>
                    <h1>Chapter 1</h1>
                    <p>Content of chapter 1.</p>
                    <h1>Chapter 2</h1>
                    <p>Content of chapter 2.</p>
                    <h1>Chapter 3</h1>
                    <p>Content of chapter 3.</p>
                </body>
            </html>
            """,
            [
                Document(page_content="Chapter 1", metadata={"Header 1": "Chapter 1"}),
                Document(
                    page_content="Content of chapter 1.",
                    metadata={"Header 1": "Chapter 1"},
                ),
                Document(page_content="Chapter 2", metadata={"Header 1": "Chapter 2"}),
                Document(
                    page_content="Content of chapter 2.",
                    metadata={"Header 1": "Chapter 2"},
                ),
                Document(page_content="Chapter 3", metadata={"Header 1": "Chapter 3"}),
                Document(
                    page_content="Content of chapter 3.",
                    metadata={"Header 1": "Chapter 3"},
                ),
            ],
            "Multiple headers of the same level",
        ),
        (
            # Test Case 5: Headers with no content
            [("h1", "Header 1"), ("h2", "Header 2")],
            """
            <html>
                <body>
                    <h1>Header 1</h1>
                    <h2>Header 2</h2>
                    <h1>Header 3</h1>
                </body>
            </html>
            """,
            [
                Document(page_content="Header 1", metadata={"Header 1": "Header 1"}),
                Document(
                    page_content="Header 2",
                    metadata={"Header 1": "Header 1", "Header 2": "Header 2"},
                ),
                Document(page_content="Header 3", metadata={"Header 1": "Header 3"}),
            ],
            "Headers with no associated content",
        ),
    ],
)
@pytest.mark.requires("bs4")
def test_html_header_text_splitter(
    html_header_splitter_splitter_factory: Callable[
        [list[tuple[str, str]]], HTMLHeaderTextSplitter
    ],
    headers_to_split_on: list[tuple[str, str]],
    html_input: str,
    expected_documents: list[Document],
    test_case: str,
) -> None:
    """Test the HTML header text splitter.

    Args:
        html_header_splitter_splitter_factory : Factory function to create the HTML
            header splitter.
        headers_to_split_on: List of headers to split on.
        html_input: The HTML input string to be split.
        expected_documents: List of expected Document objects.
        test_case: Description of the test case.

    Raises:
        AssertionError: If the number of documents or their content/metadata
            does not match the expected values.
    """
    splitter = html_header_splitter_splitter_factory(headers_to_split_on)
    docs = splitter.split_text(html_input)

    assert len(docs) == len(expected_documents), (
        f"Test Case '{test_case}' Failed: Number of documents mismatch. "
        f"Expected {len(expected_documents)}, got {len(docs)}."
    )
    for idx, (doc, expected) in enumerate(
        zip(docs, expected_documents, strict=False), start=1
    ):
        assert doc.page_content == expected.page_content, (
            f"Test Case '{test_case}' Failed at Document {idx}: "
            f"Content mismatch.\nExpected: {expected.page_content}"
            "\nGot: {doc.page_content}"
        )
        assert doc.metadata == expected.metadata, (
            f"Test Case '{test_case}' Failed at Document {idx}: "
            f"Metadata mismatch.\nExpected: {expected.metadata}\nGot: {doc.metadata}"
        )


@pytest.mark.parametrize(
    ("headers_to_split_on", "html_content", "expected_output", "test_case"),
    [
        (
            # Test Case A: Split on h1 and h2 with h3 in content
            [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
            """
            <!DOCTYPE html>
            <html>
            <body>
                <div>
                    <h1>Foo</h1>
                    <p>Some intro text about Foo.</p>
                    <div>
                        <h2>Bar main section</h2>
                        <p>Some intro text about Bar.</p>
                        <h3>Bar subsection 1</h3>
                        <p>Some text about the first subtopic of Bar.</p>
                        <h3>Bar subsection 2</h3>
                        <p>Some text about the second subtopic of Bar.</p>
                    </div>
                    <div>
                        <h2>Baz</h2>
                        <p>Some text about Baz</p>
                    </div>
                    <br>
                    <p>Some concluding text about Foo</p>
                </div>
            </body>
            </html>
            """,
            [
                Document(metadata={"Header 1": "Foo"}, page_content="Foo"),
                Document(
                    metadata={"Header 1": "Foo"},
                    page_content="Some intro text about Foo.",
                ),
                Document(
                    metadata={"Header 1": "Foo", "Header 2": "Bar main section"},
                    page_content="Bar main section",
                ),
                Document(
                    metadata={"Header 1": "Foo", "Header 2": "Bar main section"},
                    page_content="Some intro text about Bar.",
                ),
                Document(
                    metadata={
                        "Header 1": "Foo",
                        "Header 2": "Bar main section",
                        "Header 3": "Bar subsection 1",
                    },
                    page_content="Bar subsection 1",
                ),
                Document(
                    metadata={
                        "Header 1": "Foo",
                        "Header 2": "Bar main section",
                        "Header 3": "Bar subsection 1",
                    },
                    page_content="Some text about the first subtopic of Bar.",
                ),
                Document(
                    metadata={
                        "Header 1": "Foo",
                        "Header 2": "Bar main section",
                        "Header 3": "Bar subsection 2",
                    },
                    page_content="Bar subsection 2",
                ),
                Document(
                    metadata={
                        "Header 1": "Foo",
                        "Header 2": "Bar main section",
                        "Header 3": "Bar subsection 2",
                    },
                    page_content="Some text about the second subtopic of Bar.",
                ),
                Document(
                    metadata={"Header 1": "Foo", "Header 2": "Baz"}, page_content="Baz"
                ),
                Document(
                    metadata={"Header 1": "Foo"},
                    page_content=(
                        "Some text about Baz  \nSome concluding text about Foo"
                    ),
                ),
            ],
            "Test Case A: Split on h1, h2, and h3 with nested headers",
        ),
        (
            # Test Case B: Split on h1 only without any headers
            [("h1", "Header 1")],
            """
            <html>
                <body>
                    <p>Paragraph one.</p>
                    <p>Paragraph two.</p>
                    <p>Paragraph three.</p>
                </body>
            </html>
            """,
            [
                Document(
                    metadata={},
                    page_content="Paragraph one.  \nParagraph two.  \nParagraph three.",
                )
            ],
            "Test Case B: Split on h1 only without any headers",
        ),
    ],
)
@pytest.mark.requires("bs4")
def test_additional_html_header_text_splitter(
    html_header_splitter_splitter_factory: Callable[
        [list[tuple[str, str]]], HTMLHeaderTextSplitter
    ],
    headers_to_split_on: list[tuple[str, str]],
    html_content: str,
    expected_output: list[Document],
    test_case: str,
) -> None:
    """Test the HTML header text splitter.

    Args:
        html_header_splitter_splitter_factory: Factory function to create the HTML
            header splitter.
        headers_to_split_on: List of headers to split on.
        html_content: HTML content to be split.
        expected_output: Expected list of `Document` objects.
        test_case: Description of the test case.

    Raises:
        AssertionError: If the number of documents or their content/metadata
            does not match the expected output.
    """
    splitter = html_header_splitter_splitter_factory(headers_to_split_on)
    docs = splitter.split_text(html_content)

    assert len(docs) == len(expected_output), (
        f"{test_case} Failed: Number of documents mismatch. "
        f"Expected {len(expected_output)}, got {len(docs)}."
    )
    for idx, (doc, expected) in enumerate(
        zip(docs, expected_output, strict=False), start=1
    ):
        assert doc.page_content == expected.page_content, (
            f"{test_case} Failed at Document {idx}: "
            f"Content mismatch.\nExpected: {expected.page_content}\n"
            "Got: {doc.page_content}"
        )
        assert doc.metadata == expected.metadata, (
            f"{test_case} Failed at Document {idx}: "
            f"Metadata mismatch.\nExpected: {expected.metadata}\nGot: {doc.metadata}"
        )


@pytest.mark.parametrize(
    ("headers_to_split_on", "html_content", "expected_output", "test_case"),
    [
        (
            # Test Case C: Split on h1, h2, and h3 with no headers present
            [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
            """
            <html>
                <body>
                    <p>Just some random text without headers.</p>
                    <div>
                        <span>More text here.</span>
                    </div>
                </body>
            </html>
            """,
            [
                Document(
                    page_content="Just some random text without headers."
                    "  \nMore text here.",
                    metadata={},
                )
            ],
            "Test Case C: Split on h1, h2, and h3 without any headers",
        )
    ],
)
@pytest.mark.requires("bs4")
def test_html_no_headers_with_multiple_splitters(
    html_header_splitter_splitter_factory: Callable[
        [list[tuple[str, str]]], HTMLHeaderTextSplitter
    ],
    headers_to_split_on: list[tuple[str, str]],
    html_content: str,
    expected_output: list[Document],
    test_case: str,
) -> None:
    """Test HTML content splitting without headers using multiple splitters.

    Args:
        html_header_splitter_splitter_factory: Factory to create the HTML header
            splitter.
        headers_to_split_on: List of headers to split on.
        html_content: HTML content to be split.
        expected_output: Expected list of `Document` objects after splitting.
        test_case: Description of the test case.

    Raises:
        AssertionError: If the number of documents or their content/metadata
            does not match the expected output.
    """
    splitter = html_header_splitter_splitter_factory(headers_to_split_on)
    docs = splitter.split_text(html_content)

    assert len(docs) == len(expected_output), (
        f"{test_case} Failed: Number of documents mismatch. "
        f"Expected {len(expected_output)}, got {len(docs)}."
    )
    for idx, (doc, expected) in enumerate(
        zip(docs, expected_output, strict=False), start=1
    ):
        assert doc.page_content == expected.page_content, (
            f"{test_case} Failed at Document {idx}: "
            f"Content mismatch.\nExpected: {expected.page_content}\n"
            "Got: {doc.page_content}"
        )
        assert doc.metadata == expected.metadata, (
            f"{test_case} Failed at Document {idx}: "
            f"Metadata mismatch.\nExpected: {expected.metadata}\nGot: {doc.metadata}"
        )


def test_split_text_on_tokens() -> None:
    """Test splitting by tokens per chunk."""
    text = "foo bar baz 123"

    tokenizer = Tokenizer(
        chunk_overlap=3,
        tokens_per_chunk=7,
        decode=(lambda it: "".join(chr(i) for i in it)),
        encode=(lambda it: [ord(c) for c in it]),
    )
    output = split_text_on_tokens(text=text, tokenizer=tokenizer)
    expected_output = ["foo bar", "bar baz", "baz 123"]
    assert output == expected_output


def test_decode_returns_no_chunks() -> None:
    """Test that when decode returns only empty strings, output is empty, not ['']."""
    text = "foo bar baz 123"

    tokenizer = Tokenizer(
        chunk_overlap=3,
        tokens_per_chunk=7,
        decode=(lambda _: ""),
        encode=(lambda it: [ord(c) for c in it]),
    )
    output = split_text_on_tokens(text=text, tokenizer=tokenizer)
    expected_output: list[Any] = []
    assert output == expected_output


@pytest.mark.requires("bs4")
@pytest.mark.requires("lxml")
def test_section_aware_happy_path_splitting_based_on_header_1_2() -> None:
    # arrange
    html_string = """<!DOCTYPE html>
            <html>
            <body>
                <div>
                    <h1>Foo</h1>
                    <p>Some intro text about Foo.</p>
                    <div>
                        <h2>Bar main section</h2>
                        <p>Some intro text about Bar.</p>
                        <h3>Bar subsection 1</h3>
                        <p>Some text about the first subtopic of Bar.</p>
                        <h3>Bar subsection 2</h3>
                        <p>Some text about the second subtopic of Bar.</p>
                    </div>
                    <div>
                        <h2>Baz</h2>
                        <p>Some text about Baz</p>
                    </div>
                    <br>
                    <p>Some concluding text about Foo</p>
                </div>
            </body>
            </html>"""

    sec_splitter = HTMLSectionSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
    )

    docs = sec_splitter.split_text(html_string)

    assert len(docs) == 3
    assert docs[0].metadata["Header 1"] == "Foo"
    assert docs[0].page_content == "Foo \n Some intro text about Foo."

    assert docs[1].page_content == (
        "Bar main section \n Some intro text about Bar. \n "
        "Bar subsection 1 \n Some text about the first subtopic of Bar. \n "
        "Bar subsection 2 \n Some text about the second subtopic of Bar."
    )
    assert docs[1].metadata["Header 2"] == "Bar main section"

    assert (
        docs[2].page_content
        == "Baz \n Some text about Baz \n \n \n Some concluding text about Foo"
    )
    # Baz \n Some text about Baz \n \n \n Some concluding text about Foo
    # Baz \n Some text about Baz \n \n Some concluding text about Foo
    assert docs[2].metadata["Header 2"] == "Baz"


@pytest.mark.requires("bs4")
@pytest.mark.requires("lxml")
def test_happy_path_splitting_based_on_header_with_font_size() -> None:
    # arrange
    html_string = """<!DOCTYPE html>
            <html>
            <body>
                <div>
                    <span style="font-size: 22px">Foo</span>
                    <p>Some intro text about Foo.</p>
                    <div>
                        <h2>Bar main section</h2>
                        <p>Some intro text about Bar.</p>
                        <h3>Bar subsection 1</h3>
                        <p>Some text about the first subtopic of Bar.</p>
                        <h3>Bar subsection 2</h3>
                        <p>Some text about the second subtopic of Bar.</p>
                    </div>
                    <div>
                        <h2>Baz</h2>
                        <p>Some text about Baz</p>
                    </div>
                    <br>
                    <p>Some concluding text about Foo</p>
                </div>
            </body>
            </html>"""

    sec_splitter = HTMLSectionSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
    )

    docs = sec_splitter.split_text(html_string)

    assert len(docs) == 3
    assert docs[0].page_content == "Foo \n Some intro text about Foo."
    assert docs[0].metadata["Header 1"] == "Foo"

    assert docs[1].page_content == (
        "Bar main section \n Some intro text about Bar. \n "
        "Bar subsection 1 \n Some text about the first subtopic of Bar. \n "
        "Bar subsection 2 \n Some text about the second subtopic of Bar."
    )
    assert docs[1].metadata["Header 2"] == "Bar main section"

    assert docs[2].page_content == (
        "Baz \n Some text about Baz \n \n \n Some concluding text about Foo"
    )
    assert docs[2].metadata["Header 2"] == "Baz"


@pytest.mark.requires("bs4")
@pytest.mark.requires("lxml")
def test_happy_path_splitting_based_on_header_with_whitespace_chars() -> None:
    # arrange
    html_string = """<!DOCTYPE html>
            <html>
            <body>
                <div>
                    <span style="font-size: 22px">\nFoo </span>
                    <p>Some intro text about Foo.</p>
                    <div>
                        <h2>Bar main section</h2>
                        <p>Some intro text about Bar.</p>
                        <h3>Bar subsection 1</h3>
                        <p>Some text about the first subtopic of Bar.</p>
                        <h3>Bar subsection 2</h3>
                        <p>Some text about the second subtopic of Bar.</p>
                    </div>
                    <div>
                        <h2>Baz</h2>
                        <p>Some text about Baz</p>
                    </div>
                    <br>
                    <p>Some concluding text about Foo</p>
                </div>
            </body>
            </html>"""

    sec_splitter = HTMLSectionSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
    )

    docs = sec_splitter.split_text(html_string)

    assert len(docs) == 3
    assert docs[0].page_content == "Foo  \n Some intro text about Foo."
    assert docs[0].metadata["Header 1"] == "Foo"

    assert docs[1].page_content == (
        "Bar main section \n Some intro text about Bar. \n "
        "Bar subsection 1 \n Some text about the first subtopic of Bar. \n "
        "Bar subsection 2 \n Some text about the second subtopic of Bar."
    )
    assert docs[1].metadata["Header 2"] == "Bar main section"

    assert docs[2].page_content == (
        "Baz \n Some text about Baz \n \n \n Some concluding text about Foo"
    )
    assert docs[2].metadata["Header 2"] == "Baz"


@pytest.mark.requires("bs4")
@pytest.mark.requires("lxml")
def test_happy_path_splitting_with_duplicate_header_tag() -> None:
    # arrange
    html_string = """<!DOCTYPE html>
        <html>
        <body>
            <div>
                <h1>Foo</h1>
                <p>Some intro text about Foo.</p>
                <div>
                    <h2>Bar main section</h2>
                    <p>Some intro text about Bar.</p>
                    <h3>Bar subsection 1</h3>
                    <p>Some text about the first subtopic of Bar.</p>
                    <h3>Bar subsection 2</h3>
                    <p>Some text about the second subtopic of Bar.</p>
                </div>
                <div>
                    <h2>Foo</h2>
                    <p>Some text about Baz</p>
                </div>
                <h1>Foo</h1>
                <br>
                <p>Some concluding text about Foo</p>
            </div>
        </body>
        </html>"""

    sec_splitter = HTMLSectionSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
    )

    docs = sec_splitter.split_text(html_string)

    assert len(docs) == 4
    assert docs[0].page_content == "Foo \n Some intro text about Foo."
    assert docs[0].metadata["Header 1"] == "Foo"

    assert docs[1].page_content == (
        "Bar main section \n Some intro text about Bar. \n "
        "Bar subsection 1 \n Some text about the first subtopic of Bar. \n "
        "Bar subsection 2 \n Some text about the second subtopic of Bar."
    )
    assert docs[1].metadata["Header 2"] == "Bar main section"

    assert docs[2].page_content == "Foo \n Some text about Baz"
    assert docs[2].metadata["Header 2"] == "Foo"

    assert docs[3].page_content == "Foo \n \n Some concluding text about Foo"
    assert docs[3].metadata["Header 1"] == "Foo"


def test_split_json() -> None:
    """Test json text splitter."""
    max_chunk = 800
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk)

    def random_val() -> str:
        return "".join(random.choices(string.ascii_letters, k=random.randint(4, 12)))

    test_data: Any = {
        "val0": random_val(),
        "val1": {f"val1{i}": random_val() for i in range(100)},
    }
    test_data["val1"]["val16"] = {f"val16{i}": random_val() for i in range(100)}

    # uses create_docs and split_text
    docs = splitter.create_documents(texts=[test_data])

    output = [len(doc.page_content) < max_chunk * 1.05 for doc in docs]
    expected_output = [True for doc in docs]
    assert output == expected_output


def test_split_json_with_lists() -> None:
    """Test json text splitter with list conversion."""
    max_chunk = 800
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk)

    def random_val() -> str:
        return "".join(random.choices(string.ascii_letters, k=random.randint(4, 12)))

    test_data: Any = {
        "val0": random_val(),
        "val1": {f"val1{i}": random_val() for i in range(100)},
    }
    test_data["val1"]["val16"] = {f"val16{i}": random_val() for i in range(100)}

    test_data_list: Any = {"testPreprocessing": [test_data]}

    # test text splitter
    texts = splitter.split_text(json_data=test_data)
    texts_list = splitter.split_text(json_data=test_data_list, convert_lists=True)

    assert len(texts_list) >= len(texts)


def test_split_json_many_calls() -> None:
    x = {"a": 1, "b": 2}
    y = {"c": 3, "d": 4}

    splitter = RecursiveJsonSplitter()
    chunk0 = splitter.split_json(x)
    assert chunk0 == [{"a": 1, "b": 2}]

    chunk1 = splitter.split_json(y)
    assert chunk1 == [{"c": 3, "d": 4}]

    # chunk0 is now altered by creating chunk1
    assert chunk0 == [{"a": 1, "b": 2}]

    chunk0_output = [{"a": 1, "b": 2}]
    chunk1_output = [{"c": 3, "d": 4}]

    assert chunk0 == chunk0_output
    assert chunk1 == chunk1_output


def test_powershell_code_splitter_short_code() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.POWERSHELL, chunk_size=60, chunk_overlap=0
    )
    code = """
# Check if a file exists
$filePath = "C:\\temp\\file.txt"
if (Test-Path $filePath) {
    # File exists
} else {
    # File does not exist
}
    """

    chunks = splitter.split_text(code)
    assert chunks == [
        '# Check if a file exists\n$filePath = "C:\\temp\\file.txt"',
        "if (Test-Path $filePath) {\n    # File exists\n} else {",
        "# File does not exist\n}",
    ]


def test_powershell_code_splitter_longer_code() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.POWERSHELL, chunk_size=60, chunk_overlap=0
    )
    code = """
# Get a list of all processes and export to CSV
$processes = Get-Process
$processes | Export-Csv -Path "C:\\temp\\processes.csv" -NoTypeInformation

# Read the CSV file and display its content
$csvContent = Import-Csv -Path "C:\\temp\\processes.csv"
$csvContent | ForEach-Object {
    $_.ProcessName
}

# End of script
    """

    chunks = splitter.split_text(code)
    assert chunks == [
        "# Get a list of all processes and export to CSV",
        "$processes = Get-Process",
        '$processes | Export-Csv -Path "C:\\temp\\processes.csv"',
        "-NoTypeInformation",
        "# Read the CSV file and display its content",
        '$csvContent = Import-Csv -Path "C:\\temp\\processes.csv"',
        "$csvContent | ForEach-Object {\n    $_.ProcessName\n}",
        "# End of script",
    ]


FAKE_VISUALBASIC6_TEXT = """
Option Explicit

Public Function SumTwoIntegers(ByVal a As Integer, ByVal b As Integer) As Integer
    SumTwoIntegers = a + b
End Function

Public Sub Main()
    Dim i As Integer
    Dim limit As Integer

    i = 0
    limit = 50

    While i < limit
        i = SumTwoIntegers(i, 1)

        If i = limit \\ 2 Then
            MsgBox "Halfway there! i = " & i
        End If
    Wend

    MsgBox "Done! Final value of i: " & i
End Sub
"""


def test_visualbasic6_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.VISUALBASIC6,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
    )
    chunks = splitter.split_text(FAKE_VISUALBASIC6_TEXT)

    assert chunks == [
        "Option Explicit",
        "Public Function",
        "SumTwoIntegers(",
        "ByVal",
        "a As Integer,",
        "ByVal b As",
        "Integer) As",
        "Integer",
        "SumTwoIntegers",
        "= a + b",
        "End Function",
        "Public Sub",
        "Main()",
        "Dim i As",
        "Integer",
        "Dim limit",
        "As Integer",
        "i = 0",
        "limit = 50",
        "While i <",
        "limit",
        "i =",
        "SumTwoIntegers(",
        "i,",
        "1)",
        "If i =",
        "limit \\ 2 Then",
        'MsgBox "Halfway',
        'there! i = " &',
        "i",
        "End If",
        "Wend",
        "MsgBox",
        '"Done! Final',
        'value of i: " &',
        "i",
        "End Sub",
    ]


def custom_iframe_extractor(iframe_tag: Tag) -> str:
    iframe_src = iframe_tag.get("src", "")
    return f"[iframe:{iframe_src}]({iframe_src})"


@pytest.mark.requires("bs4")
def test_html_splitter_with_custom_extractor() -> None:
    """Test HTML splitting with a custom extractor."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is an iframe:</p>
    <iframe src="http://example.com"></iframe>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            custom_handlers={"iframe": custom_iframe_extractor},
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is an iframe: "
            "[iframe:http://example.com](http://example.com)",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_href_links() -> None:
    """Test HTML splitting with href links."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is a link to <a href="http://example.com">example.com</a></p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            preserve_links=True,
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is a link to [example.com](http://example.com)",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_nested_elements() -> None:
    """Test HTML splitting with nested elements."""
    html_content = """
    <h1>Main Section</h1>
    <div>
        <p>Some text here.</p>
        <div>
            <p>Nested content.</p>
        </div>
    </div>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")], max_chunk_size=1000
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="Some text here. Nested content.",
            metadata={"Header 1": "Main Section"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_preserved_elements() -> None:
    """Test HTML splitter with preserved elements.

    Test HTML splitting with preserved elements like <table>, <ul> with low chunk
    size.
    """
    html_content = """
    <h1>Section 1</h1>
    <table>
        <tr><td>Row 1</td></tr>
        <tr><td>Row 2</td></tr>
    </table>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["table", "ul"],
            max_chunk_size=50,  # Deliberately low to test preservation
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="Row 1 Row 2 Item 1 Item 2",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected  # Shouldn't split the table or ul


@pytest.mark.requires("bs4")
def test_html_splitter_with_nested_preserved_elements() -> None:
    """Test HTML splitter with preserved elements nested in containers.

    Test that preserved elements are correctly preserved even when they are
    nested inside other container elements like <section> or <article>.
    This is a regression test for issue #31569
    """
    html_content = """
    <article>
        <h1>Section 1</h1>
        <section>
            <p>Some context about the data:</p>
            <table>
                <tr><td>Col1</td><td>Col2</td></tr>
                <tr><td>Data1</td><td>Data2</td></tr>
            </table>
            <p>Conclusion about data.</p>
        </section>
    </article>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["table"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    # The table should be preserved in the output
    assert len(documents) == 1
    content = documents[0].page_content
    # Check that the table structure is maintained (not flattened)
    assert "Col1" in content
    assert "Col2" in content
    assert "Data1" in content
    assert "Data2" in content
    # Check metadata
    assert documents[0].metadata == {"Header 1": "Section 1"}


@pytest.mark.requires("bs4")
def test_html_splitter_with_nested_div_preserved() -> None:
    """Test HTML splitter preserving nested div elements.

    Nested div elements should be preserved when specified in elements_to_preserve
    """
    html_content = """
    <div>
        <h1>Header</h1>
        <p>outer text</p>
        <div>inner div content</div>
        <p>more outer text</p>
    </div>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["div"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    assert len(documents) == 1
    content = documents[0].page_content
    # The inner div content should be preserved
    assert "inner div content" in content
    assert "outer text" in content
    assert "more outer text" in content


@pytest.mark.requires("bs4")
def test_html_splitter_preserve_nested_in_paragraph() -> None:
    """Test preserving deeply nested elements (code inside paragraph).

    tests the case where a preserved element (<code>) is nested
    inside a non-container element (<p>)
    """
    html_content = "<p>before <code>KEEP_THIS</code> after</p>"
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[],
            elements_to_preserve=["code"],
        )
    documents = splitter.split_text(html_content)

    assert len(documents) == 1
    content = documents[0].page_content
    # All text should be preserved
    assert "before" in content
    assert "KEEP_THIS" in content
    assert "after" in content


@pytest.mark.requires("bs4")
def test_html_splitter_with_no_further_splits() -> None:
    """Test HTML splitting that requires no further splits beyond sections."""
    html_content = """
    <h1>Section 1</h1>
    <p>Some content here.</p>
    <h1>Section 2</h1>
    <p>More content here.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")], max_chunk_size=1000
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(page_content="Some content here.", metadata={"Header 1": "Section 1"}),
        Document(page_content="More content here.", metadata={"Header 1": "Section 2"}),
    ]

    assert documents == expected  # No further splits, just sections


@pytest.mark.requires("bs4")
def test_html_splitter_with_small_chunk_size() -> None:
    """Test HTML splitting with a very small chunk size to validate chunking."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some long text that should be split into multiple chunks due to the
    small chunk size.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")], max_chunk_size=20, chunk_overlap=5
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(page_content="This is some long", metadata={"Header 1": "Section 1"}),
        Document(page_content="long text that", metadata={"Header 1": "Section 1"}),
        Document(page_content="that should be", metadata={"Header 1": "Section 1"}),
        Document(page_content="be split into", metadata={"Header 1": "Section 1"}),
        Document(page_content="into multiple", metadata={"Header 1": "Section 1"}),
        Document(page_content="chunks due to the", metadata={"Header 1": "Section 1"}),
        Document(page_content="the small chunk", metadata={"Header 1": "Section 1"}),
        Document(page_content="size.", metadata={"Header 1": "Section 1"}),
    ]

    assert documents == expected  # Should split into multiple chunks


@pytest.mark.requires("bs4")
def test_html_splitter_with_denylist_tags() -> None:
    """Test HTML splitting with denylist tag filtering."""
    html_content = """
    <h1>Section 1</h1>
    <p>This paragraph should be kept.</p>
    <span>This span should be removed.</span>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            denylist_tags=["span"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This paragraph should be kept.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_external_metadata() -> None:
    """Test HTML splitting with external metadata integration."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some content.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            external_metadata={"source": "example.com"},
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some content.",
            metadata={"Header 1": "Section 1", "source": "example.com"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_text_normalization() -> None:
    """Test HTML splitting with text normalization."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some TEXT that should be normalized!</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            normalize_text=True,
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="this is some text that should be normalized",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_allowlist_tags() -> None:
    """Test HTML splitting with allowlist tag filtering."""
    html_content = """
    <h1>Section 1</h1>
    <p>This paragraph should be kept.</p>
    <span>This span should be kept.</span>
    <div>This div should be removed.</div>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            allowlist_tags=["p", "span"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This paragraph should be kept. This span should be kept.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_mixed_preserve_and_filter() -> None:
    """Test HTML splitting with both preserved elements and denylist tags."""
    html_content = """
    <h1>Section 1</h1>
    <table>
        <tr>
            <td>Keep this table</td>
            <td>Cell contents kept, span removed
                <span>This span should be removed.</span>
            </td>
        </tr>
    </table>
    <p>This paragraph should be kept.</p>
    <span>This span should be removed.</span>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["table"],
            denylist_tags=["span"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="Keep this table Cell contents kept, span removed"
            " This paragraph should be kept.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_no_headers() -> None:
    """Test HTML splitting when there are no headers to split on."""
    html_content = """
    <p>This is content without any headers.</p>
    <p>It should still produce a valid document.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is content without any headers. It should still produce"
            " a valid document.",
            metadata={},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_media_preservation() -> None:
    """Test HTML splitter with media preservation.

    Test HTML splitting with media elements preserved and converted to Markdown-like
    links.
    """
    html_content = """
    <h1>Section 1</h1>
    <p>This is an image:</p>
    <img src="http://example.com/image.png" />
    <p>This is a video:</p>
    <video src="http://example.com/video.mp4"></video>
    <p>This is audio:</p>
    <audio src="http://example.com/audio.mp3"></audio>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            preserve_images=True,
            preserve_videos=True,
            preserve_audio=True,
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is an image: ![image:http://example.com/image.png]"
            "(http://example.com/image.png) "
            "This is a video: ![video:http://example.com/video.mp4]"
            "(http://example.com/video.mp4) "
            "This is audio: ![audio:http://example.com/audio.mp3]"
            "(http://example.com/audio.mp3)",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_true() -> None:
    """Test HTML splitting with keep_separator=True."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
            keep_separator=True,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content=". This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_false() -> None:
    """Test HTML splitting with keep_separator=False."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
            keep_separator=False,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content="This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_start() -> None:
    """Test HTML splitting with keep_separator="start"."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
            keep_separator="start",
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content=". This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_end() -> None:
    """Test HTML splitting with keep_separator="end"."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
            keep_separator="end",
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text.",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content="This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_default() -> None:
    """Test HTML splitting with keep_separator not set."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content=". This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


def test_character_text_splitter_discard_regex_separator_on_merge() -> None:
    """Test that regex lookahead separator is not re-inserted when merging."""
    text = "SCE191 First chunk. SCE103 Second chunk."
    splitter = CharacterTextSplitter(
        separator=r"(?=SCE\d{3})",
        is_separator_regex=True,
        chunk_size=200,
        chunk_overlap=0,
        keep_separator=False,
    )
    output = splitter.split_text(text)
    assert output == ["SCE191 First chunk. SCE103 Second chunk."]


@pytest.mark.parametrize(
    ("separator", "is_regex", "text", "chunk_size", "expected"),
    [
        # 1) regex lookaround & split happens
        #   "abcmiddef" split by "(?<=mid)" → ["abcmid","def"], chunk_size=5 keeps both
        (r"(?<=mid)", True, "abcmiddef", 5, ["abcmid", "def"]),
        # 2) regex lookaround & no split
        #   chunk_size=100 merges back into ["abcmiddef"]
        (r"(?<=mid)", True, "abcmiddef", 100, ["abcmiddef"]),
        # 3) literal separator & split happens
        #   split on "mid" → ["abc","def"], chunk_size=3 keeps both
        ("mid", False, "abcmiddef", 3, ["abc", "def"]),
        # 4) literal separator & no split
        #   chunk_size=100 merges back into ["abcmiddef"]
        ("mid", False, "abcmiddef", 100, ["abcmiddef"]),
    ],
)
def test_character_text_splitter_chunk_size_effect(
    separator: str,
    *,
    is_regex: bool,
    text: str,
    chunk_size: int,
    expected: list[str],
) -> None:
    splitter = CharacterTextSplitter(
        separator=separator,
        is_separator_regex=is_regex,
        chunk_size=chunk_size,
        chunk_overlap=0,
        keep_separator=False,
    )
    assert splitter.split_text(text) == expected
