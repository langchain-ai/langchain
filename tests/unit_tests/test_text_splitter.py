"""Test text splitting functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    CppCodeTextSplitter,
    GolangCodeTextSplitter,
    JavaCodeTextSplitter,
    JavaScriptCodeTextSplitter,
    PhpCodeTextSplitter,
    ProtoCodeTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    RSTCodeTextSplitter,
    RubyCodeTextSplitter,
    RustCodeTextSplitter,
    ScalaCodeTextSplitter,
    SwiftCodeTextSplitter,
)


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


def test_character_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(ValueError):
        CharacterTextSplitter(chunk_size=2, chunk_overlap=4)


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
        "a weird",
        "text to",
        "write, but",
        "gotta test",
        "the",
        "splittingg",
        "ggg",
        "some how.",
        "Bye!\n\n-H.",
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


CHUNK_SIZE = 16


def test_python_code_splitter() -> None:
    splitter = PythonCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
def two_sum(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    
    return []
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_golang_code_splitter() -> None:
    splitter = GolangCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
func twoSum(nums []int, target int) []int {
    numMap := make(map[int]int)

    for i, num := range nums {
        complement := target - num
        if index, ok := numMap[complement]; ok {
            return []int{index, i}
        }
        numMap[num] = i
    }

    return []int{}
}
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_rst_code_splitter() -> None:
    splitter = RSTCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
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

Code Block
----------

.. code-block:: python

   print("Hello, world!")

Conclusion
----------

This concludes the sample document.
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_proto_file_splitter() -> None:
    splitter = ProtoCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
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
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_javascript_code_splitter() -> None:
    splitter = JavaScriptCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
function twoSum(nums, target) {
    let numMap = new Map();

    for (let i = 0; i < nums.length; i++) {
        let complement = target - nums[i];
        if (numMap.has(complement)) {
            return [numMap.get(complement), i];
        }
        numMap.set(nums[i], i);
    }

    return [];
}
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_java_code_splitter() -> None:
    splitter = JavaCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
public class TwoSum {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> numMap = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (numMap.containsKey(complement)) {
                return new int[] { numMap.get(complement), i };
            }
            numMap.put(nums[i], i);
        }

        return new int[] {};
    }
}
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_cpp_code_splitter() -> None:
    splitter = CppCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
#include <vector>
#include <unordered_map>

class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        std::unordered_map<int, int> numMap;

        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            if (numMap.count(complement)) {
                return { numMap[complement], i };
            }
            numMap[nums[i]] = i;
        }

        return {};
    }
};
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_scala_code_splitter() -> None:
    splitter = ScalaCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
object TwoSum {
  def twoSum(nums: Array[Int], target: Int): Array[Int] = {
    val numMap = scala.collection.mutable.Map[Int, Int]()

    for (i <- nums.indices) {
      val complement = target - nums(i)
      if (numMap.contains(complement)) {
        return Array(numMap(complement), i)
      }
      numMap(nums(i)) = i
    }

    Array.empty[Int]
  }
}
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_ruby_code_splitter() -> None:
    splitter = RubyCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
def two_sum(nums, target)
  num_map = {}

  nums.each_with_index do |num, i|
    complement = target - num
    if num_map.key?(complement)
      return [num_map[complement], i]
    end
    num_map[num] = i
  end

  []
end
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_php_code_splitter() -> None:
    splitter = PhpCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
function twoSum($nums, $target) {
    $numMap = [];

    foreach ($nums as $i => $num) {
        $complement = $target - $num;
        if (isset($numMap[$complement])) {
            return [$numMap[$complement], $i];
        }
        $numMap[$num] = $i;
    }

    return [];
}
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_swift_code_splitter() -> None:
    splitter = SwiftCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
    var numMap = [Int: Int]()

    for (i, num) in nums.enumerated() {
        let complement = target - num
        if let index = numMap[complement] {
            return [index, i]
        }
        numMap[num] = i
    }

    return []
}
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE


def test_rust_code_splitter() -> None:
    splitter = RustCodeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    code = """
fn two_sum(nums: &[i32], target: i32) -> Vec<i32> {
    let mut num_map = std::collections::HashMap::new();

    for (i, &num) in nums.iter().enumerate() {
        let complement = target - num;
        if let Some(&index) = num_map.get(&complement) {
            return vec![index as i32, i as i32];
        }
        num_map.insert(num, i);
    }

    vec![]
}
    """
    chunks = splitter.split_text(code)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE
