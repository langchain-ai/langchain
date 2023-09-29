def test_version_available() -> None:
    """Test that a version is available."""
    from langserve import __version__  # noqa: F401
