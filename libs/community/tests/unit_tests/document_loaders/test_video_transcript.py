import unittest

from langchain_community.document_loaders import AzureWhisperVideoLoader
from langchain_community.document_loaders.video_transcript import _format_start_time


class TestVideoTranscriptHelpers(unittest.TestCase):
    def test_format_start_time(self) -> None:
        self.assertEqual(_format_start_time(3600), "01:00:00")
        self.assertEqual(_format_start_time(60), "01:00")
        self.assertEqual(_format_start_time(0), "0:00")
        self.assertEqual(_format_start_time("test"), "")
        self.assertEqual(_format_start_time(3600.5), "01:00:00")
        self.assertEqual(_format_start_time(3600.6), "01:00:01")
    
    def test_class_loading(self) -> None:
        loader = AzureWhisperVideoLoader(
            video_path="test",
            api_key="test",
            api_version="test",
            azure_endpoint="test",
            deployment_id="test",
        )
        self.assertIsInstance(loader, AzureWhisperVideoLoader)
 