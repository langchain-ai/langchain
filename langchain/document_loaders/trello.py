"""Loader that loads cards from Trello"""
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class TrelloLoader(BaseLoader):
    """Trello loader. Reads all cards from a Trello board.

    Args:
        api_key (str): Trello API key.
        api_token (str): Trello API token.
        board_name (str): The name of the Trello board.
        include_card_name (bool): Whether to include the name of the card in the document. Defaults to False.
        include_comments (bool): Whether to include the comments on the card in the document. Defaults to False.
        include_checklist (bool): Whether to include the checklist on the card in the document. Defaults to False.
        card_filter (str, optional): Use "closed" / "open". Defaults to "all".
        extra_metadata (tuple[str]): List of additional metadata fields to include as document metadata. Defaults to ["due_date", "labels", "list", "is_closed"].

    """

    def __init__(
        self,
        api_key: str,
        api_token: str,
        board_name: str,
        include_card_name: bool = True,
        include_comments: bool = True,
        include_checklist: bool = True,
        card_filter: str = "all",
        extra_metadata: tuple[str, ...] = ("due_date", "labels", "list", "is_closed"),
    ):
        """Initialize Trello loader."""
        self.api_key = api_key
        self.api_token = api_token
        self.board_name = board_name
        self.include_card_name = include_card_name
        self.include_comments = include_comments
        self.include_checklist = include_checklist
        self.extra_metadata = extra_metadata
        self.card_filter = card_filter

    def load(self) -> List[Document]:
        """Loads all cards from the specified Trello board.
        You can filter the cards, metadata and text included by using the optional parameters.

         Returns:
            A list of documents, one for each card in the board.
        """

        try:
            from trello import TrelloClient  # type: ignore
        except ImportError as ex:
            raise ImportError(
                "Could not import trello python package. "
                "Please install it with `pip install py-trello`."
            ) from ex
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError as ex:
            raise ImportError(
                "`beautifulsoup4` package not found, please run"
                " `pip install beautifulsoup4`"
            ) from ex

        docs: List[Document] = []
        client = TrelloClient(api_key=self.api_key, token=self.api_token)

        # Find the board with the matching name
        board = next(
            (b for b in client.list_boards() if b.name == self.board_name), None
        )
        if not board:
            raise ValueError(f"Board `{self.board_name}` not found.")

        # Create a dictionary with the list IDs as keys and the list names as values
        list_dict = {list_item.id: list_item.name for list_item in board.list_lists()}

        # Get Cards on the board
        cards = board.get_cards(card_filter=self.card_filter)
        for card in cards:
            text_content = ""
            if self.include_card_name:
                text_content = card.name + "\n"
            description = card.description.strip()
            if description:
                text_content += BeautifulSoup(card.description, "lxml").get_text()

            if self.include_checklist:
                # Get all the checklit items on the card
                items = []
                for checklist in card.checklists:
                    if checklist.items:
                        items.extend(
                            [
                                f"{item['name']}:{item['state']}"
                                for item in checklist.items
                            ]
                        )
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
            if "is_closed" in self.extra_metadata:
                metadata["is_closed"] = card.is_closed
            if "due_date" in self.extra_metadata:
                metadata["due_date"] = card.due_date

            doc = Document(page_content=text_content, metadata=metadata)
            docs.append(doc)
        return docs
