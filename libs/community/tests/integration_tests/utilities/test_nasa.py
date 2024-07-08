"""Integration test for NASA API Wrapper."""

from langchain_community.utilities.nasa import NasaAPIWrapper


def test_media_search() -> None:
    """Test for NASA Image and Video Library media search"""
    nasa = NasaAPIWrapper()
    query = '{"q": "saturn", + "year_start": "2002", "year_end": "2010", "page": 2}'
    output = nasa.run("search_media", query)
    assert output is not None
    assert "collection" in output


def test_get_media_metadata_manifest() -> None:
    """Test for retrieving media metadata manifest from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run("get_media_metadata_manifest", "2022_0707_Recientemente")
    assert output is not None


def test_get_media_metadata_location() -> None:
    """Test for retrieving media metadata location from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run("get_media_metadata_location", "as11-40-5874")
    assert output is not None


def test_get_video_captions_location() -> None:
    """Test for retrieving video captions location from NASA Image and Video Library"""
    nasa = NasaAPIWrapper()
    output = nasa.run("get_video_captions_location", "172_ISS-Slosh.sr")
    assert output is not None
