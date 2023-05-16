"""Loader that loads cards from Trello"""
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class TrelloLoader(BaseLoader):
    """Trello loader. Reads all cards from a Trello board.

    Args:
        api_key (str): Trello API key.
        api_token (str): Trello API token.
    """

    def __init__(self, api_key: str, api_token: str):
        """Initialize Trello loader."""
        self.api_key = api_key
        self.api_token = api_token

    def load(
        self,
        board_name: str,
        card_filter: Optional[str] = "all",
        include_card_name: Optional[bool] = True,
        include_comments: Optional[bool] = True,
        include_checklist: Optional[bool] = True,
        extra_metadata: Optional[List[str]] = [
            "due_date", "labels", "list", "is_closed"],
    ) -> List[Document]:
        """Loads all cards from the specified Trello board.
        You can filter the cards, metadata and text included by using the optional parameters.

        Args:
            board_name (str): The name of the Trello board.
            card_filter (str, optional): Use "closed" / "open". Defaults to "all".
            include_card_name (bool, optional): Whether to include the name of the card in the document. Defaults to False.
            include_comments (bool, optional): Whether to include the comments on the card in the document. Defaults to False.
            include_checklist (bool, optional): Whether to include the checklist on the card in the document. Defaults to False.
            extra_metadata (List[str], optional): List of additional metadata fields to include as document metadata. Defaults to ["due_date", "labels", "list", "is_closed"].
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
        board = next((b for b in client.list_boards()
                     if b.name == board_name), None)
        if board:
            board_id = board.id
        else:
            raise ValueError(f"Board `{board_name}` not found.")
        
        # Get the lists in the board
        lists = board.list_lists()

        # Create a dictionary with the list IDs as keys and the list names as values
        list_dict = {}
        for list_item in lists:
            list_dict[list_item.id] = list_item.name
            
        # Get Cards on the board
        board = client.get_board(board_id)
        cards = board.get_cards(card_filter=card_filter)
        for card in cards:
            text_content = ""
            if include_card_name:
                text_content = card.name + "\n"
            description = card.description.strip()
            if description:
                text_content += BeautifulSoup(card.description,
                                              "lxml").get_text()

            if include_checklist:
                # Get all the checklit items on the card
                items = []
                for checklist in card.checklists:
                    items.extend(
                        [f"{item['name']}:{item['state']}" for item in checklist.items])
                    text_content += f"\n{checklist.name}\n" + "\n".join(items)

            if include_comments:
                # Get all the comments on the card
                comments = [BeautifulSoup(comment['data']['text'], "lxml").get_text()
                            for comment in card.comments]
                text_content += "Comments:" + "\n".join(comments)

            # Default metadata fields
            metadata = {
                "title": card.name,
                "id": card.id,
                "url": card.url,
            }

            # Extra metadata fields. Card object is not subscriptable.
            if "labels" in extra_metadata:
                metadata["labels"] = [label.name for label in card.labels]
            if "list" in extra_metadata:
                if card.list_id in list_dict:
                    metadata["list"] = list_dict[card.list_id]
            if "is_close" in extra_metadata:
                metadata["is_closed"] = card.is_closed
            if "due_date" in extra_metadata:
                metadata["due_date"] = card.due_date

            doc = Document(page_content=text_content, metadata=metadata)
            docs.append(doc)
        return docs
