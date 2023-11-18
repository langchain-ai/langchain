"""Integration test for NASA API Wrapper."""
import pytest

from langchain.utilities.nasa import NasaAPIWrapper

# for 'output' and 'assert' statements, reference PROMPT.py examples
def test_media_search() -> None:
    """Test for NASA Image and Video Library media search"""
    nasa = NasaAPIWrapper()
    output = nasa.run()
    assert "" in output

def test_get_media_metadata_manifest() -> None:
    """Test for retrieving media metadata manifest from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run()
    assert "" in output

def test_get_media_metadata_location() -> None:
    """Test for retrieving media metadata location from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run()
    assert "" in output

def test_get_video_captions_location() -> None:
    """Test for retrieving video captions location from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run()
    assert "" in output