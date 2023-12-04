from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader
from langchain.utils import get_from_env

if TYPE_CHECKING:
    from trello import Board, Card, TrelloClient


class TrelloLoader(BaseLoader):
    """Load cards from a `Trello` board."""

    def __init__(
        self,
        client: TrelloClient,
        board_name: str,
        *,
        include_card_name: bool = True,
        include_comments: bool = True,
        include_checklist: bool = True,
        card_filter: Literal["closed", "open", "all"] = "all",
        extra_metadata: Tuple[str, ...] = ("due_date", "labels", "list", "closed"),
    ):
        """Initialize Trello loader.

        Args:
            client: Trello API client.
            board_name: The name of the Trello board.
            include_card_name: Whether to include the name of the card in the document.
            include_comments: Whether to include the comments on the card in the
                document.
            include_checklist: Whether to include the checklist on the card in the
                document.
            card_filter: Filter on card status. Valid values are "closed", "open",
                "all".
            extra_metadata: List of additional metadata fields to include as document
                metadata.Valid values are "due_date", "labels", "list", "closed".

        """
        self.client = client
        self.board_name = board_name
        self.include_card_name = include_card_name
        self.include_comments = include_comments
        self.include_checklist = include_checklist
        self.extra_metadata = extra_metadata
        self.card_filter = card_filter

    @classmethod
    def from_credentials(
        cls,
        board_name: str,
        *,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> TrelloLoader:
        """Convenience constructor that builds TrelloClient init param for you.

        Args:
            board_name: The name of the Trello board.
            api_key: Trello API key. Can also be specified as environment variable
                TRELLO_API_KEY.
            token: Trello token. Can also be specified as environment variable
                TRELLO_TOKEN.
            include_card_name: Whether to include the name of the card in the document.
            include_comments: Whether to include the comments on the card in the
                document.
            include_checklist: Whether to include the checklist on the card in the
                document.
            card_filter: Filter on card status. Valid values are "closed", "open",
                "all".
            extra_metadata: List of additional metadata fields to include as document
                metadata.Valid values are "due_date", "labels", "list", "closed".
        """

        try:
            from trello import TrelloClient  # type: ignore
        except ImportError as ex:
            raise ImportError(
                "Could not import trello python package. "
                "Please install it with `pip install py-trello`."
            ) from ex
        api_key = api_key or get_from_env("api_key", "TRELLO_API_KEY")
        token = token or get_from_env("token", "TRELLO_TOKEN")
        client = TrelloClient(api_key=api_key, token=token)
        return cls(client, board_name, **kwargs)

    def load(self) -> List[Document]:
        """Loads all cards from the specified Trello board.

        You can filter the cards, metadata and text included by using the optional
            parameters.

         Returns:
            A list of documents, one for each card in the board.
        """
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError as ex:
            raise ImportError(
                "`beautifulsoup4` package not found, please run"
                " `pip install beautifulsoup4`"
            ) from ex

        board = self._get_board()
        # Create a dictionary with the list IDs as keys and the list names as values
        list_dict = {list_item.id: list_item.name for list_item in board.list_lists()}
        # Get Cards on the board
        cards = board.get_cards(card_filter=self.card_filter)
        return [self._card_to_doc(card, list_dict) for card in cards]

    def _get_board(self) -> Board:
        # Find the first board with a matching name
        board = next(
            (b for b in self.client.list_boards() if b.name == self.board_name), None
        )
        if not board:
            raise ValueError(f"Board `{self.board_name}` not found.")
        return board

    def _card_to_doc(self, card: Card, list_dict: dict) -> Document:
        from bs4 import BeautifulSoup  # type: ignore

        text_content = ""
        if self.include_card_name:
            text_content = card.name + "\n"
        if card.description.strip():
            text_content += BeautifulSoup(card.description, "lxml").get_text()
        if self.include_checklist:
            # Get all the checklist items on the card
            for checklist in card.checklists:
                if checklist.items:
                    items = [
                        f"{item['name']}:{item['state']}" for item in checklist.items
                    ]
                    text_content += f"\n{checklist.name}\n" + "\n".join(items)

        if self.include_comments:
            # Get all the comments on the card
            comments = [
                BeautifulSoup(comment["data"]["text"], "lxml").get_text()
                for comment in card.comments
            ]
            text_content += "Comments:" + "\n".join(comments)

        # Default metadata fields
        metadata = {
            "title": card.name,
            "id": card.id,
            "url": card.url,
        }

        # Extra metadata fields. Card object is not subscriptable.
        if "labels" in self.extra_metadata:
            metadata["labels"] = [label.name for label in card.labels]
        if "list" in self.extra_metadata:
            if card.list_id in list_dict:
                metadata["list"] = list_dict[card.list_id]
        if "closed" in self.extra_metadata:
            metadata["closed"] = card.closed
        if "due_date" in self.extra_metadata:
            metadata["due_date"] = card.due_date

        return Document(page_content=text_content, metadata=metadata)
