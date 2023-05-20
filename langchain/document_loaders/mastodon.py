"""Mastodon document loader."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    import mastodon


def _dependable_mastodon_import() -> mastodon:
    try:
        import mastodon
    except ImportError:
        raise ValueError(
            "Mastodon.py package not found, "
            + "please install it with `pip install Mastodon.py`"
        )
    return mastodon


class MastodonTootsLoader(BaseLoader):
    """Mastodon toots loader.
    Read toots of user's Mastodon handle.

    Attributes:
        access_token:
          An access token to use if the request is made as a Mastodon app.
        api_base_url:
          The base URL of the Mastodon instance API for the loader to talk to.
        mastodon_accounts:
          A list of Mastodon accounts to pull toots for.
        number_toots:
          The integer number of toots pulled during load for each listed account.
        exclude_replies:
          A boolean indicating whether replies are excluded from the pulls
    """

    def __init__(
        self,
        mastodon_accounts: Sequence[str],
        number_toots: Optional[int] = 100,
        exclude_replies: bool = False,
        access_token: Optional[str] = None,
        api_base_url: str = "https://mastodon.social",
    ):
        """Mastodon toots loader.

        Read toots of a list of Mastodon users.

        Args:
            mastodon_accounts: The list of Mastodon accounts to query.
            number_toots: How many toots to pull for each account.
            exclude_replies: Whether or not to exclude reply toots from the load.
            access_token: An access token if toots are loaded as a Mastodon app.
            api_base_url: A Mastodon API base URL to talk to, if not using the default.
        """
        self.access_token = access_token
        self.api_base_url = api_base_url
        self.mastodon_accounts = mastodon_accounts
        self.number_toots = number_toots
        self.exclude_replies = exclude_replies

    def load(self) -> List[Document]:
        """Load toots into documents."""
        mastodon = _dependable_mastodon_import()
        api = mastodon.Mastodon(
            access_token=self.access_token, api_base_url=self.api_base_url
        )

        results: List[Document] = []
        for account in self.mastodon_accounts:
            user = api.account_lookup(account)
            toots = api.account_statuses(
                user.id,
                only_media=False,
                pinned=False,
                exclude_replies=self.exclude_replies,
                exclude_reblogs=True,
                limit=self.number_toots,
            )
            docs = self._format_toots(toots, user)
            results.extend(docs)
        return results

    def _format_toots(
        self, toots: List[Dict[str, Any]], user_info: dict
    ) -> Iterable[Document]:
        """Format toots into documents.

        Adding user info, and selected toot fields into the metadata.
        """
        for toot in toots:
            metadata = {
                "created_at": toot["created_at"],
                "user_info": user_info,
                "is_reply": toot["in_reply_to_id"] is not None,
            }
            yield Document(
                page_content=toot["content"],
                metadata=metadata,
            )
