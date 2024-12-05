---
pagination_prev: contributing/how_to/integrations/index
pagination_next: contributing/how_to/integrations/publish
---
# How to implement and test a retriever integration

In this guide, we'll implement and test a custom [retriever](/docs/concepts/retrievers) that you have integrated with LangChain.

For testing, we will rely on the `langchain-tests` dependency we added in the previous [package creation guide](/docs/contributing/how_to/integrations/package).

## Implementation

Let's say you're building a simple integration package that provides a `ToyRetriever`
retriever integration for LangChain. Here's a simple example of what your project
structure might look like:

```plaintext
langchain-parrot-link/
├── langchain_parrot_link/
│   ├── __init__.py
│   └── retrievers.py
├── tests/
│   └── integration_tests
|       ├── __init__.py
|       └── test_retrievers.py
├── pyproject.toml
└── README.md
```

In this first step, we will implement the `retrievers.py` file

import CustomRetrieverIntro from '/docs/how_to/_custom_retriever_intro.mdx';

<CustomRetrieverIntro />

<details>
    <summary>retrievers.py</summary>
```python title="langchain_parrot_link/retrievers.py"
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class ParrotRetriever(BaseRetriever):
    parrot_name: str
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k = kwargs.get("k", self.k)
        return [Document(page_content=f"{self.parrot_name} says: {query}")] * k
```
</details>

:::tip

The `ParrotRetriever` from this guide is tested
against the standard unit and integration tests in the LangChain Github repository.
You can always use this as a starting point [here](https://github.com/langchain-ai/langchain/blob/master/libs/standard-tests/tests/unit_tests/test_basic_retriever.py).

:::

## Testing



### 1. Create Your Retriever Class

```python
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

class MyCustomRetriever(BaseRetriever):
    """Custom retriever implementation."""
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Core implementation of retrieving relevant documents."""
        # Your implementation here
        pass

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async implementation of retrieving relevant documents."""
        # Your async implementation here
        pass
```

### 2. Required Testing

All retrievers must include the following tests:

#### Basic Functionality Tests
```python
def test_get_relevant_documents():
    retriever = MyCustomRetriever()
    docs = retriever.get_relevant_documents("test query")
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)

@pytest.mark.asyncio
async def test_aget_relevant_documents():
    retriever = MyCustomRetriever()
    docs = await retriever.aget_relevant_documents("test query")
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
```

#### Edge Cases
- Empty query handling
- Special character handling
- Long query handling
- Rate limiting (if applicable)
- Error handling

### 3. Documentation Requirements

Your retriever should include:

1. Class docstring with:
   - General description
   - Required dependencies
   - Example usage
   - Parameters explanation

2. Integration documentation file:
   - Installation instructions
   - Basic usage example
   - Advanced configuration
   - Common issues and solutions

### 4. Best Practices

1. **Error Handling**
   - Implement proper error handling for API calls
   - Provide meaningful error messages
   - Handle rate limits gracefully

2. **Performance**
   - Implement caching when appropriate
   - Use batch operations where possible
   - Consider implementing both sync and async methods

3. **Configuration**
   - Use environment variables for sensitive data
   - Provide sensible defaults
   - Allow for customization of key parameters

4. **Type Hints**
   - Use proper type hints throughout your code
   - Document expected types in docstrings

## Example Implementation

Here's a minimal example of a custom retriever:

```python
from typing import List
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

class SimpleKeywordRetriever(BaseRetriever):
    """A simple retriever that matches documents based on keywords."""
    
    documents: List[Document]  # Store your documents here
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Return documents that contain the query string."""
        return [
            doc for doc in self.documents 
            if query.lower() in doc.page_content.lower()
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self._get_relevant_documents(query, run_manager=run_manager)
```

## Submission Checklist

- [ ] Implemented base retriever interface
- [ ] Added comprehensive tests
- [ ] Included proper documentation
- [ ] Added type hints
- [ ] Handled error cases
- [ ] Implemented both sync and async methods
- [ ] Added example usage
- [ ] Followed code style guidelines
- [ ] Added requirements.txt or setup.py updates

## Getting Help

If you need help while implementing your retriever:
1. Check existing retriever implementations for reference
2. Open a discussion in the GitHub repository
3. Ask in the LangChain Discord community

Remember to follow the existing patterns in the codebase and maintain consistency with other retrievers.
