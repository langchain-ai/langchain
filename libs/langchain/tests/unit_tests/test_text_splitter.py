"""Test text splitting functionality."""
import re
from typing import List

import pytest

from langchain.docstore.document import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    SplittedText,
)

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
    expected_output = ["foo", "bar", "baz", "a", "a a"]
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


@pytest.mark.parametrize(
    "separator, is_separator_regex", [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_keep_separator_regex(
    separator: str, is_separator_regex: bool
) -> None:
    """Test splitting by characters while keeping the separator
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
    expected_output = ["foo", ".", "bar", ".", "baz", ".", "123"]
    assert output == expected_output


@pytest.mark.parametrize(
    "separator, is_separator_regex", [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_discard_separator_regex(
    separator: str, is_separator_regex: bool
) -> None:
    """Test splitting by characters discarding the separator
    that is a regex special character."""
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


def test_character_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(ValueError):
        CharacterTextSplitter(chunk_size=2, chunk_overlap=4)


def test_merge_splits() -> None:
    """Test merging splits with a given separator."""
    splitter = CharacterTextSplitter(separator=" ", chunk_size=9, chunk_overlap=2)
    splits = [SplittedText(text=text) for text in ["foo", "bar", "baz"]]
    expected_output = ["foobarbaz"]
    output = splitter._merge_splits(splits)
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


def test_create_documents_with_start_index() -> None:
    """Test create documents method."""
    texts = ["foo bar baz 123"]
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=7, chunk_overlap=3, add_start_index=True
    )
    docs = splitter.create_documents(texts)
    expected_docs = [
        Document(page_content="foo bar", metadata={"start_index": 0}),
        Document(page_content="bar baz", metadata={"start_index": 4}),
        Document(page_content="baz 123", metadata={"start_index": 8}),
    ]
    assert docs == expected_docs


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

    assert output == ["....5X", "..3Y", "...4X", "....5Y", "..."]


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


def __test_iterative_text_splitter(chunk_size: int, keep_separator: bool) -> List[str]:
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
This is a weird to write, but gotta test the splittingggg some how.

Bye!\n\n-H."""
    chunk_size = 10
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=1, force_chunk_size=True
    )
    output = splitter.split_text(text)
    expected_output = [
        "Hi.\n\nI'm",
        "Harrison.",
        "How?",
        "Are? You?",
        "Okay then",
        "f f f f.",
        "This is a",
        "a weird to",
        "write,",
        "but gotta",
        "test the",
        "splittingg",
        "gg",
        "some how.",
        "Bye!",
        "-H.",
    ]
    assert output == expected_output
    assert all(len(chunk) <= chunk_size for chunk in output)


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
    assert splits == [
        "class Foo:\n\n    def bar():",
        "def foo():\n\ndef",
        "testing_func():\n\ndef bar():",
    ]


CHUNK_SIZE = 16


def __test_language_splitter(
    language: Language,
    code: str,
    expected: List[str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = 0,
) -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    actual = splitter.split_text(code)

    assert expected == actual


def test_python_code_splitter() -> None:
    code = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
    """
    expected = [
        "def",
        "hello_world():",
        'print("Hello,',
        'World!")\n\n# Call',
        "the function",
        "hello_world()",
    ]
    __test_language_splitter(Language.PYTHON, code, expected)


def test_golang_code_splitter() -> None:
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
    expected = [
        "package main",
        'import "fmt"',
        "func",
        "helloWorld() {",
        'fmt.Println("Hello,',
        'World!")\n}',
        "func main() {",
        "helloWorld()",
        "}",
    ]
    __test_language_splitter(Language.GO, code, expected)


def test_rst_code_splitter() -> None:
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
    expected = [
        "Sample Document",
        "===============",
        "Section",
        "-------\n\nThis",
        "is the content",
        "of the section.",
        "Lists\n-----\n\n-",
        "Item 1\n- Item 2",
        "- Item 3",
        "Comment\n*******",
        "Not a comment",
        ".. This is a",
        "comment",
    ]
    __test_language_splitter(Language.RST, code, expected)

    # Special test for special characters
    code = "harry\n***\nbabylon is"
    expected = ["harry\n***", "babylon is"]
    __test_language_splitter(Language.RST, code, expected)


def test_proto_file_splitter() -> None:
    code = """
syntax = "proto3";

package example;

message Person {
    string name = 1;
    int32 age = 2;
    repeated string hobbies = 3;
}
    """
    expected = [
        "syntax =",
        '"proto3";',
        "package example;",
        "message Person",
        "{\n    string",
        "name = 1;",
        "int32 age = 2;",
        "repeated",
        "string hobbies =",
        "3;\n}",
    ]
    __test_language_splitter(Language.PROTO, code, expected)


def test_javascript_code_splitter() -> None:
    code = """
function helloWorld() {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
    """
    expected = [
        "function",
        "helloWorld() {",
        'console.log("Hello,',
        'World!");\n}\n\n//',
        "Call the",
        "function",
        "helloWorld();",
    ]
    __test_language_splitter(Language.JS, code, expected)


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
        "HelloWorld.\nDATA",
        "DIVISION.",
        "WORKING-STORAGE",
        "SECTION.\n01",
        "GREETING",
        "PIC",
        "X(12)   VALUE",
        "'Hello, World!'.",
        "PROCEDURE",
        "DIVISION.",
        "DISPLAY",
        "GREETING.\nSTOP",
        "RUN.",
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
        'console.log("Hello,',
        'World!");\n}\n\n//',
        "Call the",
        "function",
        "helloWorld();",
    ]


def test_java_code_splitter() -> None:
    code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
    """
    expected = [
        "public class",
        "HelloWorld {",
        "public",
        "static void",
        "main(String[]",
        "args) {",
        'System.out.println("Hello,',
        'World!");\n    }',
        "}",
    ]
    __test_language_splitter(Language.JAVA, code, expected)


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
        "Array<String>) {",
        'println("Hello,',
        'World!")',
        "}\n    }",
        "}",
    ]


def test_csharp_code_splitter() -> None:
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
    expected = [
        "using System;",
        "class Program\n{",
        "static void",
        "Main()\n    {",
        "int age",
        "= 30; // Change",
        "the age value as",
        "needed",
        "//",
        "Categorize the",
        "age without any",
        "console output",
        "if (age",
        "< 18)\n        {",
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
        "}\n    }\n}",
    ]
    __test_language_splitter(Language.CSHARP, code, expected)


def test_cpp_code_splitter() -> None:
    code = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
    """
    expected = [
        "#include",
        "<iostream>\n\nint",
        "main() {",
        "std::cout <<",
        '"Hello, World!"',
        "<< std::endl;",
        "return 0;\n}",
    ]
    __test_language_splitter(Language.CPP, code, expected)


def test_scala_code_splitter() -> None:
    code = """
object HelloWorld {
  def main(args: Array[String]): Unit = {
    println("Hello, World!")
  }
}
    """
    expected = [
        "object",
        "HelloWorld {",
        "def main(args:",
        "Array[String]):",
        "Unit = {",
        'println("Hello,',
        'World!")\n  }\n}',
    ]
    __test_language_splitter(Language.SCALA, code, expected)


def test_ruby_code_splitter() -> None:
    code = """
def hello_world
  puts "Hello, World!"
end

hello_world
    """
    expected = ["def hello_world", 'puts "Hello,', 'World!"\nend', "hello_world"]
    __test_language_splitter(Language.RUBY, code, expected)


def test_php_code_splitter() -> None:
    code = """
<?php
function hello_world() {
    echo "Hello, World!";
}

hello_world();
?>
    """
    expected = [
        "<?php\nfunction",
        "hello_world() {",
        'echo "Hello,',
        'World!";\n}',
        "hello_world();",
        "?>",
    ]
    __test_language_splitter(Language.PHP, code, expected)


def test_swift_code_splitter() -> None:
    code = """
func helloWorld() {
    print("Hello, World!")
}

helloWorld()
    """
    expected = [
        "func",
        "helloWorld() {",
        'print("Hello,',
        'World!")\n}',
        "helloWorld()",
    ]
    __test_language_splitter(Language.SWIFT, code, expected)


def test_rust_code_splitter() -> None:
    code = """
fn main() {
    println!("Hello, World!");
}
    """
    expected = ["fn main() {", 'println!("Hello,', 'World!");\n}']
    __test_language_splitter(Language.RUST, code, expected)


def test_markdown_code_splitter() -> None:
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
    expected = [
        "# Sample",
        "Document\n\n##",
        "Section\n\nThis is",
        "the content of",
        "the section.\n\n##",
        "Lists\n\n- Item 1",
        "- Item 2\n- Item",
        "3\n\n###",
        "Horizontal lines",
        "***********",
        "____________",
        "-------------------",
        "#### Code",
        "blocks\n```\nThis",
        "is a code block",
        "# sample code",
        "a = 1\nb = 2\n```",
    ]
    __test_language_splitter(Language.MARKDOWN, code, expected)

    # Special test for special characters
    code = "harry\n***\nbabylon is"
    expected = ["harry\n***", "babylon is"]
    __test_language_splitter(Language.MARKDOWN, code, expected)


def test_latex_code_splitter() -> None:
    code = """
Hi Harrison!
\\chapter{1}
"""
    expected = ["Hi Harrison!", "\\chapter{1}"]
    __test_language_splitter(Language.LATEX, code, expected)


def test_html_code_splitter() -> None:
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
    expected = [
        "<h1>Sample Document</h1>\n    <h2>Section</h2>\n        <p",
        'id="1234">Reference content.</p>\n\n    <h2>Lists</h2>',
        "<ul>\n            <li>Item 1</li>",
        "<li>Item 2</li>\n            <li>Item 3</li>\n        </ul>",
        '<h3>A block</h3>\n            <div class="amazing">',
        "<p>Some text</p>\n                <p>Some",
        "more text</p>\n            </div>",
    ]
    __test_language_splitter(Language.HTML, code, expected, chunk_size=60)


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


def test_solidity_code_splitter() -> None:
    code = """pragma solidity ^0.8.20;
  contract HelloWorld {
    function add(uint a, uint b) pure public returns(uint) {
      return  a + b;
    }
  }
  """
    expected = [
        "pragma solidity",
        "^0.8.20;",
        "contract",
        "HelloWorld {",
        "function",
        "add(uint a, uint",
        "b) pure public",
        "returns(uint) {",
        "return  a",
        "+ b;\n    }\n  }",
    ]
    __test_language_splitter(Language.SOL, code, expected)
