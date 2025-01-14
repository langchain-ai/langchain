# 🦜✂️ LangChain Text Splitters

[![Downloads](https://static.pepy.tech/badge/langchain_text_splitters/month)](https://pepy.tech/project/langchain_text_splitters)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Install

```bash
pip install langchain-text-splitters
```

## What is it?

LangChain Text Splitters contains utilities for splitting into chunks a wide variety of text documents.

For full documentation see the [API reference](https://python.langchain.com/api_reference/text_splitters/index.html)
and the [Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) module in the main docs.

## 📕 Releases & Versioning

`langchain-text-splitters` is currently on version `0.0.x`.

Minor version increases will occur for:

- Breaking changes for any public interfaces NOT marked `beta`

Patch version increases will occur for:

- Bug fixes
- New features
- Any changes to private interfaces
- Any changes to `beta` features

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://python.langchain.com/docs/contributing/).


## Writing Test Case Guidelines

To ensure high-quality and maintainable testing, we follow a structured approach. Below is a step-by-step guideline for writing test cases effectively:

---

### Example: `test_character_text_splitter_keep_separator_regex`

#### **1. Unique Identifier**
- Use a descriptive method name in `snake_case` format.  
  **Example:** `test_character_text_splitter_keep_separator_regex`.

#### **2. Purpose**
- Clearly state the functionality or edge case being tested. Include this as a docstring after the test method definition.  
  **Example:**  
  > *To test the behavior of `CharacterTextSplitter` when a separator is a regex special character and needs to be retained in the output chunks.*

#### **3. Testing Steps**
Each test should include the following details:

- **a. Specified States:**  
  Clearly define the inputs, initial configurations, and state of the system.  
  **Example:**  
  ```plaintext
  Initial text: "foo.bar.baz.123"  
  Separator: "." (escaped or plain based on `is_separator_regex`)  
  Chunk size: 1  
  Chunk overlap: 0  
  Keep separator: True  

- **b. Messages/Operations:**
  List the specific operations or methods invoked in the test.
  **Example:**
  
  1. Initialize `CharacterTextSplitter` with the given configurations.  
  2. Call `splitter.split_text(text)`.
 

- **c. Exceptions (Optional):**
  Mention any expected exceptions, if applicable.

- **d. External Conditions (Optional):**
  Specify any external dependencies or conditions needed for the test.
  **Example:** Requires Python's `re` module for `regex-based splitting`.

- **e. Supplementary Information:**
  State the **expected output** or any additional details to help understand the test.
  **Example:**
  Expected result: `["foo", ".bar", ".baz", ".123"]`

  
#### **4. Clarity and Modularity**
  Each test case should focus on testing one specific behavior or edge case.
  The test setup, operations, and assertions should be concise and easy to follow.
