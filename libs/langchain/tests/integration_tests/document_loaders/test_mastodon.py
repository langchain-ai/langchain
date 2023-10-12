"""Tests for the Mastodon toots loader"""
from langchain.document_loaders import MastodonTootsLoader


def test_mastodon_toots_loader() -> None:
    """Test Mastodon toots loader with an external query."""
    # Query the Mastodon CEO's account
    loader = MastodonTootsLoader(
        mastodon_accounts=["@Gargron@mastodon.social"], number_toots=1
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["user_info"]["id"] == 1
