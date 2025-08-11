"""Integration tests for OpenAI Batch API functionality.

These tests require a valid OpenAI API key and will make actual API calls.
They are designed to test the complete end-to-end batch processing workflow.
"""

import os
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.batch import BatchError

# Skip all tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set, skipping integration tests",
)


class TestBatchAPIIntegration:
    """Integration tests for OpenAI Batch API functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=50,  # Keep responses short for faster processing
        )

    @pytest.mark.scheduled
    def test_batch_create_and_retrieve_small_batch(self):
        """Test end-to-end batch processing with a small batch."""
        # Create a small batch of simple questions
        messages_list = [
            [HumanMessage(content="What is 2+2? Answer with just the number.")],
            [
                HumanMessage(
                    content=(
                        "What is the capital of France? Answer with just the city name."
                    )
                )
            ],
        ]

        # Create batch job
        batch_id = self.llm.batch_create(
            messages_list=messages_list,
            description="Integration test batch - small",
            metadata={"test_type": "integration", "batch_size": "small"},
        )

        assert isinstance(batch_id, str)
        assert batch_id.startswith("batch_")

        # Retrieve results (this will poll until completion)
        # Note: This may take several minutes for real batch processing
        results = self.llm.batch_retrieve(
            batch_id=batch_id,
            poll_interval=30.0,  # Poll every 30 seconds
            timeout=1800.0,  # 30 minute timeout
        )

        # Verify results
        assert len(results) == 2
        assert all(isinstance(result, ChatResult) for result in results)
        assert all(len(result.generations) == 1 for result in results)
        assert all(
            isinstance(result.generations[0].message, AIMessage) for result in results
        )

        # Check that we got reasonable responses
        response1 = results[0].generations[0].message.content.strip()
        response2 = results[1].generations[0].message.content.strip()

        # Basic sanity checks (responses should contain expected content)
        assert "4" in response1 or "four" in response1.lower()
        assert "paris" in response2.lower()

    @pytest.mark.scheduled
    def test_batch_method_with_batch_api_true(self):
        """Test the batch() method with use_batch_api=True."""
        inputs = [
            [HumanMessage(content="Count to 3. Answer with just: 1, 2, 3")],
            [HumanMessage(content="What color is the sky? Answer with just: blue")],
        ]

        # Use batch API mode
        results = self.llm.batch(
            inputs, use_batch_api=True, poll_interval=30.0, timeout=1800.0
        )

        # Verify results
        assert len(results) == 2
        assert all(isinstance(result, AIMessage) for result in results)
        assert all(isinstance(result.content, str) for result in results)

        # Basic sanity checks
        response1 = results[0].content.strip().lower()
        response2 = results[1].content.strip().lower()

        assert any(char in response1 for char in ["1", "2", "3"])
        assert "blue" in response2

    @pytest.mark.scheduled
    def test_batch_method_comparison(self):
        """Test that batch API and standard batch produce similar results."""
        inputs = [[HumanMessage(content="What is 1+1? Answer with just the number.")]]

        # Test standard batch processing
        standard_results = self.llm.batch(inputs, use_batch_api=False)

        # Test batch API processing
        batch_api_results = self.llm.batch(
            inputs, use_batch_api=True, poll_interval=30.0, timeout=1800.0
        )

        # Both should return similar structure
        assert len(standard_results) == len(batch_api_results) == 1
        assert isinstance(standard_results[0], AIMessage)
        assert isinstance(batch_api_results[0], AIMessage)

        # Both should contain reasonable answers
        standard_content = standard_results[0].content.strip()
        batch_content = batch_api_results[0].content.strip()

        assert "2" in standard_content or "two" in standard_content.lower()
        assert "2" in batch_content or "two" in batch_content.lower()

    @pytest.mark.scheduled
    def test_batch_with_different_parameters(self):
        """Test batch processing with different model parameters."""
        messages_list = [
            [HumanMessage(content="Write a haiku about coding. Keep it short.")]
        ]

        # Create batch with specific parameters
        batch_id = self.llm.batch_create(
            messages_list=messages_list,
            description="Integration test - parameters",
            metadata={"test_type": "parameters"},
            temperature=0.8,  # Higher temperature for creativity
            max_tokens=100,  # More tokens for haiku
        )

        results = self.llm.batch_retrieve(
            batch_id=batch_id, poll_interval=30.0, timeout=1800.0
        )

        assert len(results) == 1
        result_content = results[0].generations[0].message.content

        # Should have some content (haiku)
        assert len(result_content.strip()) > 10
        # Haikus typically have line breaks
        assert "\n" in result_content or len(result_content.split()) >= 5

    @pytest.mark.scheduled
    def test_batch_with_system_message(self):
        """Test batch processing with system messages."""
        from langchain_core.messages import SystemMessage

        messages_list = [
            [
                SystemMessage(
                    content="You are a helpful math tutor. Answer concisely."
                ),
                HumanMessage(content="What is 5 * 6?"),
            ]
        ]

        batch_id = self.llm.batch_create(
            messages_list=messages_list, description="Integration test - system message"
        )

        results = self.llm.batch_retrieve(
            batch_id=batch_id, poll_interval=30.0, timeout=1800.0
        )

        assert len(results) == 1
        result_content = results[0].generations[0].message.content.strip()

        # Should contain the answer
        assert "30" in result_content or "thirty" in result_content.lower()

    @pytest.mark.scheduled
    def test_batch_error_handling_invalid_model(self):
        """Test error handling with invalid model parameters."""
        # Create a ChatOpenAI instance with an invalid model
        invalid_llm = ChatOpenAI(model="invalid-model-name-12345", temperature=0.1)

        messages_list = [[HumanMessage(content="Hello")]]

        # This should fail during batch creation or processing
        with pytest.raises(BatchError):
            batch_id = invalid_llm.batch_create(messages_list=messages_list)
            # If batch creation succeeds, retrieval should fail
            invalid_llm.batch_retrieve(batch_id, timeout=300.0)

    def test_batch_input_conversion(self):
        """Test batch processing with various input formats."""
        # Test with string inputs (should be converted to HumanMessage)
        inputs = [
            "What is the largest planet? Answer with just the planet name.",
            [
                HumanMessage(
                    content=(
                        "What is the smallest planet? Answer with just the planet name."
                    )
                )
            ],
        ]

        results = self.llm.batch(
            inputs, use_batch_api=True, poll_interval=30.0, timeout=1800.0
        )

        assert len(results) == 2
        assert all(isinstance(result, AIMessage) for result in results)

        # Check for reasonable responses
        response1 = results[0].content.strip().lower()
        response2 = results[1].content.strip().lower()

        assert "jupiter" in response1
        assert "mercury" in response2

    @pytest.mark.scheduled
    def test_empty_batch_handling(self):
        """Test handling of empty batch inputs."""
        # Empty inputs should return empty results
        results = self.llm.batch([], use_batch_api=True)
        assert results == []

        # Empty messages list should also work
        batch_id = self.llm.batch_create(messages_list=[])
        results = self.llm.batch_retrieve(batch_id, timeout=300.0)
        assert results == []

    @pytest.mark.scheduled
    def test_batch_metadata_preservation(self):
        """Test that batch metadata is properly handled."""
        messages_list = [[HumanMessage(content="Say 'test successful'")]]

        metadata = {
            "test_name": "metadata_test",
            "user_id": "test_user_123",
            "experiment": "batch_api_integration",
        }

        # Create batch with metadata
        batch_id = self.llm.batch_create(
            messages_list=messages_list,
            description="Metadata preservation test",
            metadata=metadata,
        )

        # Retrieve results
        results = self.llm.batch_retrieve(batch_id, timeout=1800.0)

        assert len(results) == 1
        result_content = results[0].generations[0].message.content.strip().lower()
        assert "test successful" in result_content


class TestBatchAPIEdgeCases:
    """Test edge cases and error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=50)

    @pytest.mark.scheduled
    def test_batch_with_very_short_timeout(self):
        """Test batch processing with very short timeout."""
        messages_list = [[HumanMessage(content="Hello")]]

        batch_id = self.llm.batch_create(messages_list=messages_list)

        # Try to retrieve with very short timeout (should timeout)
        with pytest.raises(BatchError, match="timed out"):
            self.llm.batch_retrieve(
                batch_id=batch_id,
                poll_interval=1.0,
                timeout=5.0,  # Very short timeout
            )

    def test_batch_retrieve_invalid_batch_id(self):
        """Test retrieving results with invalid batch ID."""
        with pytest.raises(BatchError):
            self.llm.batch_retrieve("invalid_batch_id_12345", timeout=30.0)

    @pytest.mark.scheduled
    def test_batch_with_long_content(self):
        """Test batch processing with longer content."""
        long_content = "Please summarize this text: " + "This is a test sentence. " * 20

        messages_list = [[HumanMessage(content=long_content)]]

        batch_id = self.llm.batch_create(
            messages_list=messages_list,
            max_tokens=200,  # Allow more tokens for summary
        )

        results = self.llm.batch_retrieve(batch_id, timeout=1800.0)

        assert len(results) == 1
        result_content = results[0].generations[0].message.content

        # Should have some summary content
        assert len(result_content.strip()) > 10


class TestBatchAPIPerformance:
    """Performance and scalability tests."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=30,  # Keep responses short
        )

    @pytest.mark.scheduled
    def test_medium_batch_processing(self):
        """Test processing a medium-sized batch (10 requests)."""
        # Create 10 simple math questions
        messages_list = [
            [HumanMessage(content=f"What is {i} + {i}? Answer with just the number.")]
            for i in range(1, 11)
        ]

        start_time = time.time()

        batch_id = self.llm.batch_create(
            messages_list=messages_list,
            description="Medium batch test - 10 requests",
            metadata={"batch_size": "medium", "request_count": "10"},
        )

        results = self.llm.batch_retrieve(
            batch_id=batch_id,
            poll_interval=60.0,  # Poll every minute
            timeout=3600.0,  # 1 hour timeout
        )

        end_time = time.time()
        _ = end_time - start_time

        # Verify all results
        assert len(results) == 10
        assert all(isinstance(result, ChatResult) for result in results)

        # Check that we got reasonable math answers
        for i, result in enumerate(results, 1):
            content = result.generations[0].message.content.strip()
            expected_answer = str(i + i)
            assert expected_answer in content or str(i * 2) in content

        # Log processing time for analysis    @pytest.mark.scheduled

    def test_batch_vs_sequential_comparison(self):
        """Compare batch API performance vs sequential processing."""
        messages = [
            [HumanMessage(content="Count to 2. Answer: 1, 2")],
            [HumanMessage(content="Count to 3. Answer: 1, 2, 3")],
        ]

        # Test sequential processing time
        start_sequential = time.time()
        sequential_results = []
        for message_list in messages:
            result = self.llm.invoke(message_list)
            sequential_results.append(result)
        _ = time.time() - start_sequential

        # Test batch API processing time
        start_batch = time.time()
        batch_results = self.llm.batch(
            messages, use_batch_api=True, poll_interval=30.0, timeout=1800.0
        )
        _ = time.time() - start_batch

        # Verify both produce results
        assert len(sequential_results) == len(batch_results) == 2

        # Note: Batch API will typically be slower for small batches due to polling,
        # but should be more cost-effective for larger batches


# Helper functions for integration tests
def is_openai_api_available() -> bool:
    """Check if OpenAI API is available and accessible."""
    try:
        import openai

        client = openai.OpenAI()
        # Try a simple API call to verify connectivity
        client.models.list()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def openai_api_check():
    """Session-scoped fixture to check OpenAI API availability."""
    if not is_openai_api_available():
        pytest.skip("OpenAI API not available or accessible")
