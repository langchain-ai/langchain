import unittest

import pytest
from langchain_community.document_loaders.video_transcript import (
    AzureWhisperVideoParagraphLoader,
    AzureWhisperVideoSegmentLoader,
    OpenAIWhisperVideoParagraphLoader,
    OpenAIWhisperVideoSegmentLoader,
    _format_start_time,
)


class TestVideoTranscriptHelpers(unittest.TestCase):
    def test_format_start_time(self) -> None:
        self.assertEqual(_format_start_time(3600), "01:00:00")
        self.assertEqual(_format_start_time(60), "01:00")
        self.assertEqual(_format_start_time(0), "0:00")
        self.assertEqual(_format_start_time("test"), "")
        self.assertEqual(_format_start_time(3600.5), "01:00:00")
        self.assertEqual(_format_start_time(3600.6), "01:00:01")

@pytest.mark.requires("openai", "openai-whisper")
class TestAzureWhisperVideoSegmentLoader(unittest.TestCase):
    def test_class_loading(self):
        """Test the initialization of AzureWhisperVideoSegmentLoader."""
        loader = AzureWhisperVideoSegmentLoader(
            video_path="test",
            deployment_id="test",
            api_key="test",
            api_version="test",
            azure_endpoint="test",
        )
        self.assertIsInstance(loader, AzureWhisperVideoSegmentLoader)

@pytest.mark.requires("openai", "openai-whisper")
class TestAzureWhisperVideoParagraphLoader(unittest.TestCase):
    def test_class_loading(self):
        """Test the initialization of AzureWhisperVideoSegmentLoader."""
        loader = AzureWhisperVideoParagraphLoader(
            video_path="test",
            deployment_id="test",
            api_key="test",
            api_version="test",
            azure_endpoint="test",
            paragraph_sentence_size=3,
        )
        self.assertIsInstance(loader, AzureWhisperVideoParagraphLoader)

@pytest.mark.requires("openai", "openai-whisper")
class TestOpenAIWhisperVideoSegmentLoader(unittest.TestCase):
    def test_class_loading(self):
        """Test the initialization of AzureWhisperVideoSegmentLoader."""
        loader = OpenAIWhisperVideoSegmentLoader(
            video_path="test",
            api_key="test",
        )
        self.assertIsInstance(loader, OpenAIWhisperVideoSegmentLoader)


@pytest.mark.requires("openai", "openai-whisper")
class TestOpenAIWhisperVideoParagraLoader(unittest.TestCase):
    def test_class_loading(self):
        """Test the initialization of AzureWhisperVideoSegmentLoader."""
        loader = OpenAIWhisperVideoParagraphLoader(
            video_path="test", api_key="test", paragraph_sentence_size=3
        )
        self.assertIsInstance(loader, OpenAIWhisperVideoParagraphLoader)
