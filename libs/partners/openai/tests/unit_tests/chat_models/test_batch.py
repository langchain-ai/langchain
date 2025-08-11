"""Test OpenAI Batch API functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.batch import (
    BatchError,
    OpenAIBatchClient,
    OpenAIBatchProcessor,
)


class TestOpenAIBatchClient:
    """Test the OpenAIBatchClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.batch_client = OpenAIBatchClient(
            client=self.mock_client,
            poll_interval=0.1,  # Fast polling for tests
            timeout=5.0,
        )

    def test_create_batch_success(self):
        """Test successful batch creation."""
        # Mock batch creation response
        mock_batch = MagicMock()
        mock_batch.id = "batch_123"
        mock_batch.status = "validating"
        self.mock_client.batches.create.return_value = mock_batch

        batch_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ]

        batch_id = self.batch_client.create_batch(
            batch_requests=batch_requests,
            description="Test batch",
            metadata={"test": "true"},
        )

        assert batch_id == "batch_123"
        self.mock_client.batches.create.assert_called_once()

    def test_create_batch_failure(self):
        """Test batch creation failure."""
        self.mock_client.batches.create.side_effect = Exception("API Error")

        batch_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ]

        with pytest.raises(BatchError, match="Failed to create batch"):
            self.batch_client.create_batch(batch_requests=batch_requests)

    def test_poll_batch_status_completed(self):
        """Test polling until batch completion."""
        # Mock batch status progression
        mock_batch_validating = MagicMock()
        mock_batch_validating.status = "validating"

        mock_batch_in_progress = MagicMock()
        mock_batch_in_progress.status = "in_progress"

        mock_batch_completed = MagicMock()
        mock_batch_completed.status = "completed"
        mock_batch_completed.output_file_id = "file_123"

        self.mock_client.batches.retrieve.side_effect = [
            mock_batch_validating,
            mock_batch_in_progress,
            mock_batch_completed,
        ]

        result = self.batch_client.poll_batch_status("batch_123")

        assert result.status == "completed"
        assert result.output_file_id == "file_123"
        assert self.mock_client.batches.retrieve.call_count == 3

    def test_poll_batch_status_failed(self):
        """Test polling when batch fails."""
        mock_batch_failed = MagicMock()
        mock_batch_failed.status = "failed"
        mock_batch_failed.errors = [{"message": "Batch processing failed"}]

        self.mock_client.batches.retrieve.return_value = mock_batch_failed

        with pytest.raises(BatchError, match="Batch failed"):
            self.batch_client.poll_batch_status("batch_123")

    def test_poll_batch_status_timeout(self):
        """Test polling timeout."""
        mock_batch_in_progress = MagicMock()
        mock_batch_in_progress.status = "in_progress"

        self.mock_client.batches.retrieve.return_value = mock_batch_in_progress

        # Set very short timeout
        self.batch_client.timeout = 0.2

        with pytest.raises(BatchError, match="Batch polling timed out"):
            self.batch_client.poll_batch_status("batch_123")

    def test_retrieve_batch_results_success(self):
        """Test successful batch result retrieval."""
        # Mock file content
        mock_results = [
            {
                "id": "batch_req_123",
                "custom_id": "request-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Hello! How can I help you?",
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 8,
                            "total_tokens": 18,
                        },
                    },
                },
            }
        ]

        mock_file_content = "\n".join(json.dumps(result) for result in mock_results)
        self.mock_client.files.content.return_value.content = mock_file_content.encode()

        results = self.batch_client.retrieve_batch_results("file_123")

        assert len(results) == 1
        assert results[0]["custom_id"] == "request-1"
        assert results[0]["response"]["status_code"] == 200

    def test_retrieve_batch_results_failure(self):
        """Test batch result retrieval failure."""
        self.mock_client.files.content.side_effect = Exception("File not found")

        with pytest.raises(BatchError, match="Failed to retrieve batch results"):
            self.batch_client.retrieve_batch_results("file_123")


class TestOpenAIBatchProcessor:
    """Test the OpenAIBatchProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.processor = OpenAIBatchProcessor(
            client=self.mock_client,
            model="gpt-3.5-turbo",
            poll_interval=0.1,
            timeout=5.0,
        )

    def test_create_batch_success(self):
        """Test successful batch creation with message conversion."""
        # Mock batch client
        with patch.object(self.processor, "batch_client") as mock_batch_client:
            mock_batch_client.create_batch.return_value = "batch_123"

            messages_list = [
                [HumanMessage(content="What is 2+2?")],
                [HumanMessage(content="What is the capital of France?")],
            ]

            batch_id = self.processor.create_batch(
                messages_list=messages_list,
                description="Test batch",
                metadata={"test": "true"},
                temperature=0.7,
            )

            assert batch_id == "batch_123"
            mock_batch_client.create_batch.assert_called_once()

            # Verify batch requests were created correctly
            call_args = mock_batch_client.create_batch.call_args
            batch_requests = call_args[1]["batch_requests"]

            assert len(batch_requests) == 2
            assert batch_requests[0]["custom_id"] == "request-0"
            assert batch_requests[0]["body"]["model"] == "gpt-3.5-turbo"
            assert batch_requests[0]["body"]["temperature"] == 0.7
            assert batch_requests[0]["body"]["messages"][0]["role"] == "user"
            assert batch_requests[0]["body"]["messages"][0]["content"] == "What is 2+2?"

    def test_poll_batch_status_success(self):
        """Test successful batch status polling."""
        with patch.object(self.processor, "batch_client") as mock_batch_client:
            mock_batch = MagicMock()
            mock_batch.status = "completed"
            mock_batch_client.poll_batch_status.return_value = mock_batch

            result = self.processor.poll_batch_status("batch_123")

            assert result.status == "completed"
            mock_batch_client.poll_batch_status.assert_called_once_with(
                "batch_123", poll_interval=None, timeout=None
            )

    def test_retrieve_batch_results_success(self):
        """Test successful batch result retrieval and conversion."""
        # Mock batch status and results
        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file_123"

        mock_results = [
            {
                "id": "batch_req_123",
                "custom_id": "request-0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "2+2 equals 4.",
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 8,
                            "total_tokens": 18,
                        },
                    },
                },
            },
            {
                "id": "batch_req_124",
                "custom_id": "request-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "The capital of France is Paris.",
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 10,
                            "total_tokens": 22,
                        },
                    },
                },
            },
        ]

        with patch.object(self.processor, "batch_client") as mock_batch_client:
            mock_batch_client.poll_batch_status.return_value = mock_batch
            mock_batch_client.retrieve_batch_results.return_value = mock_results

            chat_results = self.processor.retrieve_batch_results("batch_123")

            assert len(chat_results) == 2

            # Check first result
            assert isinstance(chat_results[0], ChatResult)
            assert len(chat_results[0].generations) == 1
            assert isinstance(chat_results[0].generations[0].message, AIMessage)
            assert chat_results[0].generations[0].message.content == "2+2 equals 4."

            # Check second result
            assert isinstance(chat_results[1], ChatResult)
            assert len(chat_results[1].generations) == 1
            assert isinstance(chat_results[1].generations[0].message, AIMessage)
            assert (
                chat_results[1].generations[0].message.content
                == "The capital of France is Paris."
            )

    def test_retrieve_batch_results_with_errors(self):
        """Test batch result retrieval with some failed requests."""
        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file_123"

        mock_results = [
            {
                "id": "batch_req_123",
                "custom_id": "request-0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Success response",
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 8,
                            "total_tokens": 18,
                        },
                    },
                },
            },
            {
                "id": "batch_req_124",
                "custom_id": "request-1",
                "response": {
                    "status_code": 400,
                    "body": {
                        "error": {
                            "message": "Invalid request",
                            "type": "invalid_request_error",
                        }
                    },
                },
            },
        ]

        with patch.object(self.processor, "batch_client") as mock_batch_client:
            mock_batch_client.poll_batch_status.return_value = mock_batch
            mock_batch_client.retrieve_batch_results.return_value = mock_results

            with pytest.raises(BatchError, match="Batch request request-1 failed"):
                self.processor.retrieve_batch_results("batch_123")


class TestBaseChatOpenAIBatchMethods:
    """Test the batch methods added to BaseChatOpenAI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="test-key")

    @patch("langchain_openai.chat_models.batch.OpenAIBatchProcessor")
    def test_batch_create_success(self, mock_processor_class):
        """Test successful batch creation."""
        mock_processor = MagicMock()
        mock_processor.create_batch.return_value = "batch_123"
        mock_processor_class.return_value = mock_processor

        messages_list = [
            [HumanMessage(content="What is 2+2?")],
            [HumanMessage(content="What is the capital of France?")],
        ]

        batch_id = self.llm.batch_create(
            messages_list=messages_list,
            description="Test batch",
            metadata={"test": "true"},
            temperature=0.7,
        )

        assert batch_id == "batch_123"
        mock_processor_class.assert_called_once()
        mock_processor.create_batch.assert_called_once_with(
            messages_list=messages_list,
            description="Test batch",
            metadata={"test": "true"},
            temperature=0.7,
        )

    @patch("langchain_openai.chat_models.batch.OpenAIBatchProcessor")
    def test_batch_retrieve_success(self, mock_processor_class):
        """Test successful batch result retrieval."""
        mock_processor = MagicMock()
        mock_chat_results = [
            ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="2+2 equals 4."))]
            ),
            ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="The capital of France is Paris.")
                    )
                ]
            ),
        ]
        mock_processor.retrieve_batch_results.return_value = mock_chat_results
        mock_processor_class.return_value = mock_processor

        results = self.llm.batch_retrieve("batch_123", poll_interval=1.0, timeout=60.0)

        assert len(results) == 2
        assert results[0].generations[0].message.content == "2+2 equals 4."
        assert (
            results[1].generations[0].message.content
            == "The capital of France is Paris."
        )

        mock_processor.poll_batch_status.assert_called_once_with(
            batch_id="batch_123", poll_interval=1.0, timeout=60.0
        )
        mock_processor.retrieve_batch_results.assert_called_once_with("batch_123")

    @patch("langchain_openai.chat_models.batch.OpenAIBatchProcessor")
    def test_batch_method_with_batch_api_true(self, mock_processor_class):
        """Test batch method with use_batch_api=True."""
        mock_processor = MagicMock()
        mock_chat_results = [
            ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Response 1"))]
            ),
            ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Response 2"))]
            ),
        ]
        mock_processor.create_batch.return_value = "batch_123"
        mock_processor.retrieve_batch_results.return_value = mock_chat_results
        mock_processor_class.return_value = mock_processor

        inputs = [
            [HumanMessage(content="Question 1")],
            [HumanMessage(content="Question 2")],
        ]

        results = self.llm.batch(inputs, use_batch_api=True, temperature=0.5)

        assert len(results) == 2
        assert isinstance(results[0], AIMessage)
        assert results[0].content == "Response 1"
        assert isinstance(results[1], AIMessage)
        assert results[1].content == "Response 2"

        mock_processor.create_batch.assert_called_once()
        mock_processor.retrieve_batch_results.assert_called_once()

    def test_batch_method_with_batch_api_false(self):
        """Test batch method with use_batch_api=False (default behavior)."""
        inputs = [
            [HumanMessage(content="Question 1")],
            [HumanMessage(content="Question 2")],
        ]

        # Mock the parent class batch method
        with patch.object(ChatOpenAI.__bases__[0], "batch") as mock_super_batch:
            mock_super_batch.return_value = [
                AIMessage(content="Response 1"),
                AIMessage(content="Response 2"),
            ]

            results = self.llm.batch(inputs, use_batch_api=False)

            assert len(results) == 2
            mock_super_batch.assert_called_once_with(
                inputs=inputs, config=None, return_exceptions=False
            )

    def test_convert_input_to_messages_list(self):
        """Test _convert_input_to_messages helper method."""
        # Test list of messages
        messages = [HumanMessage(content="Hello")]
        result = self.llm._convert_input_to_messages(messages)
        assert result == messages

        # Test single message
        message = HumanMessage(content="Hello")
        result = self.llm._convert_input_to_messages(message)
        assert result == [message]

        # Test string input
        result = self.llm._convert_input_to_messages("Hello")
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    @patch("langchain_openai.chat_models.batch.OpenAIBatchProcessor")
    def test_batch_create_with_error_handling(self, mock_processor_class):
        """Test batch creation with error handling."""
        mock_processor = MagicMock()
        mock_processor.create_batch.side_effect = BatchError("Batch creation failed")
        mock_processor_class.return_value = mock_processor

        messages_list = [[HumanMessage(content="Test")]]

        with pytest.raises(BatchError, match="Batch creation failed"):
            self.llm.batch_create(messages_list)

    @patch("langchain_openai.chat_models.batch.OpenAIBatchProcessor")
    def test_batch_retrieve_with_error_handling(self, mock_processor_class):
        """Test batch retrieval with error handling."""
        mock_processor = MagicMock()
        mock_processor.poll_batch_status.side_effect = BatchError(
            "Batch polling failed"
        )
        mock_processor_class.return_value = mock_processor

        with pytest.raises(BatchError, match="Batch polling failed"):
            self.llm.batch_retrieve("batch_123")

    def test_batch_method_input_conversion(self):
        """Test batch method handles various input formats correctly."""
        with (
            patch.object(self.llm, "batch_create") as mock_create,
            patch.object(self.llm, "batch_retrieve") as mock_retrieve,
        ):
            mock_create.return_value = "batch_123"
            mock_retrieve.return_value = [
                ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="Response"))]
                )
            ]

            # Test with string inputs
            inputs = ["Hello world"]
            results = self.llm.batch(inputs, use_batch_api=True)

            # Verify conversion happened
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            messages_list = call_args["messages_list"]

            assert len(messages_list) == 1
            assert len(messages_list[0]) == 1
            assert isinstance(messages_list[0][0], HumanMessage)
            assert messages_list[0][0].content == "Hello world"


class TestBatchErrorHandling:
    """Test error handling scenarios."""

    def test_batch_error_creation(self):
        """Test BatchError exception creation."""
        error = BatchError("Test error message")
        assert str(error) == "Test error message"

    def test_batch_error_with_details(self):
        """Test BatchError with additional details."""
        details = {"batch_id": "batch_123", "status": "failed"}
        error = BatchError("Batch failed", details)
        assert str(error) == "Batch failed"
        assert error.args[1] == details


class TestBatchIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="test-key")

    @patch("langchain_openai.chat_models.batch.OpenAIBatchProcessor")
    def test_empty_messages_list(self, mock_processor_class):
        """Test handling of empty messages list."""
        mock_processor = MagicMock()
        mock_processor.create_batch.return_value = "batch_123"
        mock_processor.retrieve_batch_results.return_value = []
        mock_processor_class.return_value = mock_processor

        results = self.llm.batch([], use_batch_api=True)
        assert results == []

    @patch("langchain_openai.chat_models.batch.OpenAIBatchProcessor")
    def test_large_batch_processing(self, mock_processor_class):
        """Test processing of large batch."""
        mock_processor = MagicMock()
        mock_processor.create_batch.return_value = "batch_123"

        # Create mock results for large batch
        num_requests = 100
        mock_chat_results = [
            ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=f"Response {i}"))]
            )
            for i in range(num_requests)
        ]
        mock_processor.retrieve_batch_results.return_value = mock_chat_results
        mock_processor_class.return_value = mock_processor

        inputs = [[HumanMessage(content=f"Question {i}")] for i in range(num_requests)]
        results = self.llm.batch(inputs, use_batch_api=True)

        assert len(results) == num_requests
        for i, result in enumerate(results):
            assert result.content == f"Response {i}"

    @patch("langchain_openai.chat_models.batch.OpenAIBatchProcessor")
    def test_mixed_message_types(self, mock_processor_class):
        """Test batch processing with mixed message types."""
        mock_processor = MagicMock()
        mock_processor.create_batch.return_value = "batch_123"
        mock_processor.retrieve_batch_results.return_value = [
            ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Response 1"))]
            ),
            ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Response 2"))]
            ),
        ]
        mock_processor_class.return_value = mock_processor

        inputs = [
            "String input",  # Will be converted to HumanMessage
            [HumanMessage(content="Direct message list")],  # Already formatted
        ]

        results = self.llm.batch(inputs, use_batch_api=True)
        assert len(results) == 2

        # Verify the conversion happened correctly
        mock_processor.create_batch.assert_called_once()
        call_args = mock_processor.create_batch.call_args[1]
        messages_list = call_args["messages_list"]

        # First input should be converted to HumanMessage
        assert isinstance(messages_list[0][0], HumanMessage)
        assert messages_list[0][0].content == "String input"

        # Second input should remain as is
        assert isinstance(messages_list[1][0], HumanMessage)
        assert messages_list[1][0].content == "Direct message list"
