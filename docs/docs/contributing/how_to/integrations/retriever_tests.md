# Standard Tests for LangChain Retrievers

This guide outlines the standard tests that should be implemented for all LangChain retrievers.

## Test Structure

### 1. Basic Functionality Tests

```python
import pytest
from langchain.schema import Document
from your_retriever import YourRetriever

def test_basic_retrieval():
    """Test basic document retrieval functionality."""
    retriever = YourRetriever()
    query = "test query"
    docs = retriever.get_relevant_documents(query)
    
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) > 0  # Adjust if your retriever might return empty results

@pytest.mark.asyncio
async def test_async_retrieval():
    """Test async document retrieval functionality."""
    retriever = YourRetriever()
    query = "test query"
    docs = await retriever.aget_relevant_documents(query)
    
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
```

### 2. Edge Cases

```python
def test_empty_query():
    """Test behavior with empty query."""
    retriever = YourRetriever()
    docs = retriever.get_relevant_documents("")
    assert isinstance(docs, list)

def test_special_characters():
    """Test handling of special characters."""
    retriever = YourRetriever()
    special_queries = [
        "test!@#$%^&*()",
        "múltiple áccents",
        "中文测试",
        "test\nwith\nnewlines",
    ]
    for query in special_queries:
        docs = retriever.get_relevant_documents(query)
        assert isinstance(docs, list)

def test_long_query():
    """Test handling of very long queries."""
    retriever = YourRetriever()
    long_query = "test " * 1000
    docs = retriever.get_relevant_documents(long_query)
    assert isinstance(docs, list)
```

### 3. Error Handling

```python
def test_invalid_configuration():
    """Test behavior with invalid configuration."""
    with pytest.raises(ValueError):
        YourRetriever(invalid_param="invalid")

def test_connection_error():
    """Test behavior when connection fails (if applicable)."""
    retriever = YourRetriever()
    # Mock connection failure
    with pytest.raises(ConnectionError):
        retriever.get_relevant_documents("test")
```

### 4. Performance Tests (Optional)

```python
@pytest.mark.slow
def test_large_scale_retrieval():
    """Test retrieval with a large number of documents."""
    retriever = YourRetriever()
    # Test with a significant number of documents
    docs = retriever.get_relevant_documents("test")
    assert len(docs) <= YOUR_MAX_LIMIT  # If applicable

@pytest.mark.slow
def test_concurrent_requests():
    """Test handling of concurrent requests."""
    import asyncio
    
    async def run_concurrent_requests():
        retriever = YourRetriever()
        tasks = [
            retriever.aget_relevant_documents("test")
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        return results
    
    results = asyncio.run(run_concurrent_requests())
    assert len(results) == 5
```

### 5. Integration Tests

```python
def test_chain_integration():
    """Test integration with LangChain chains."""
    from langchain.chains import RetrievalQA
    from langchain.llms import FakeLLM
    
    retriever = YourRetriever()
    llm = FakeLLM()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    result = qa_chain.run("test query")
    assert isinstance(result, str)
```

## Test Configuration

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        Document(page_content="test document 1", metadata={"source": "test1"}),
        Document(page_content="test document 2", metadata={"source": "test2"}),
    ]

@pytest.fixture
def mock_retriever(sample_documents):
    """Fixture providing a retriever with sample documents."""
    retriever = YourRetriever()
    # Set up retriever with sample documents
    return retriever
```

## Running Tests

To run the tests:

```bash
# Run all tests
pytest tests/retrievers/test_your_retriever.py

# Run only fast tests
pytest tests/retrievers/test_your_retriever.py -m "not slow"

# Run with coverage
pytest tests/retrievers/test_your_retriever.py --cov=your_retriever
```

## Best Practices

1. **Isolation**: Each test should be independent and not rely on the state from other tests.

2. **Mocking**: Use mocks for external services to avoid actual API calls during testing:
   ```python
   @pytest.fixture
   def mock_api(mocker):
       return mocker.patch("your_retriever.api_client")
   ```

3. **Parametrization**: Use pytest.mark.parametrize for testing multiple scenarios:
   ```python
   @pytest.mark.parametrize("query,expected_count", [
       ("test", 1),
       ("invalid", 0),
       ("multiple words", 2),
   ])
   def test_retrieval_counts(query, expected_count):
       retriever = YourRetriever()
       docs = retriever.get_relevant_documents(query)
       assert len(docs) == expected_count
   ```

4. **Documentation**: Include docstrings in test functions explaining what they test.

5. **Coverage**: Aim for high test coverage, especially for core functionality.

## Common Pitfalls

1. Not testing error cases
2. Not testing async functionality
3. Not handling rate limits in tests
4. Missing edge cases
5. Relying on external services in unit tests

Remember to adapt these tests based on your retriever's specific functionality and requirements.
