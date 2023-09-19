import unittest
from collections import namedtuple
from typing import Any, Optional
from unittest.mock import patch

import pytest

from langchain.document_loaders.trello import TrelloLoader


def list_to_objects(dict_list: list) -> list:
    """Helper to convert dict objects."""
    return [
        namedtuple("Object", d.keys())(**d) for d in dict_list if isinstance(d, dict)
    ]


def card_list_to_objects(cards: list) -> list:
    """Helper to convert dict cards into trello weird mix of objects and dictionaries"""
    for card in cards:
        card["checklists"] = list_to_objects(card.get("checklists"))
        card["labels"] = list_to_objects(card.get("labels"))
    return list_to_objects(cards)


class MockBoard:
    """
    Defining Trello mock board internal object to use in the patched method.
    """

    def __init__(self, id: str, name: str, cards: list, lists: list):
        self.id = id
        self.name = name
        self.cards = cards
        self.lists = lists

    def get_cards(self, card_filter: Optional[str] = "") -> list:
        """We do not need to test the card-filter since is on Trello Client side."""
        return self.cards

    def list_lists(self) -> list:
        return self.lists


TRELLO_LISTS = [
    {
        "id": "5555cacbc4daa90564b34cf2",
        "name": "Publishing Considerations",
    },
    {
        "id": "5555059b74c03b3a9e362cd0",
        "name": "Backlog",
    },
    {
        "id": "555505a3427fd688c1ca5ebd",
        "name": "Selected for Milestone",
    },
    {
        "id": "555505ba95ff925f9fb1b370",
        "name": "Blocked",
    },
    {
        "id": "555505a695ff925f9fb1b13d",
        "name": "In Progress",
    },
    {
        "id": "555505bdfe380c7edc8ca1a3",
        "name": "Done",
    },
]
# Create a mock list of cards.
TRELLO_CARDS_QA = [
    {
        "id": "12350aca6952888df7975903",
        "name": "Closed Card Title",
        "description": "This is the <em>description</em> of Closed Card.",
        "closed": True,
        "labels": [],
        "due_date": "",
        "url": "https://trello.com/card/12350aca6952888df7975903",
        "list_id": "555505bdfe380c7edc8ca1a3",
        "checklists": [
            {
                "name": "Checklist 1",
                "items": [
                    {
                        "name": "Item 1",
                        "state": "pending",
                    },
                    {
                        "name": "Item 2",
                        "state": "completed",
                    },
                ],
            },
        ],
        "comments": [
            {
                "data": {
                    "text": "This is a comment on a <s>Closed</s> Card.",
                },
            },
        ],
    },
    {
        "id": "45650aca6952888df7975903",
        "name": "Card 2",
        "description": "This is the description of <strong>Card 2</strong>.",
        "closed": False,
        "labels": [{"name": "Medium"}, {"name": "Task"}],
        "due_date": "",
        "url": "https://trello.com/card/45650aca6952888df7975903",
        "list_id": "555505a695ff925f9fb1b13d",
        "checklists": [],
        "comments": [],
    },
    {
        "id": "55550aca6952888df7975903",
        "name": "Camera",
        "description": "<div></div>",
        "closed": False,
        "labels": [{"name": "Task"}],
        "due_date": "",
        "url": "https://trello.com/card/55550aca6952888df7975903",
        "list_id": "555505a3427fd688c1ca5ebd",
        "checklists": [
            {
                "name": "Tasks",
                "items": [
                    {"name": "Zoom", "state": "complete"},
                    {"name": "Follow players", "state": "complete"},
                    {
                        "name": "camera limit to stage size",
                        "state": "complete",
                    },
                    {"name": "Post Processing effects", "state": "complete"},
                    {
                        "name": "Shitch to universal render pipeline",
                        "state": "complete",
                    },
                ],
            },
        ],
        "comments": [
            {
                "data": {
                    "text": (
                        "to follow group of players use Group Camera feature of "
                        "cinemachine."
                    )
                }
            },
            {
                "data": {
                    "text": "Use 'Impulse' <s>Cinemachine</s> feature for camera shake."
                }
            },
            {"data": {"text": "depth of field with custom shader."}},
        ],
    },
]


@pytest.fixture
def mock_trello_client() -> Any:
    """Fixture that creates a mock for trello.TrelloClient."""
    # Create a mock `trello.TrelloClient` object.
    with patch("trello.TrelloClient") as mock_trello_client:
        # Create a mock list of trello list (columns in the UI).

        # The trello client returns a hierarchy mix of objects and dictionaries.
        list_objs = list_to_objects(TRELLO_LISTS)
        cards_qa_objs = card_list_to_objects(TRELLO_CARDS_QA)
        boards = [
            MockBoard("5555eaafea917522902a2a2c", "Research", [], list_objs),
            MockBoard("55559f6002dd973ad8cdbfb7", "QA", cards_qa_objs, list_objs),
        ]

        # Patch `get_boards()` method of the mock `TrelloClient` object to return the
        # mock list of boards.
        mock_trello_client.return_value.list_boards.return_value = boards
        yield mock_trello_client.return_value


@pytest.mark.usefixtures("mock_trello_client")
@pytest.mark.requires("trello", "bs4", "lxml")
class TestTrelloLoader(unittest.TestCase):
    def test_empty_board(self) -> None:
        """
        Test loading a board with no cards.
        """
        trello_loader = TrelloLoader.from_credentials(
            "Research",
            api_key="API_KEY",
            token="API_TOKEN",
        )
        documents = trello_loader.load()
        self.assertEqual(len(documents), 0, "Empty board returns an empty list.")

    def test_complete_text_and_metadata(self) -> None:
        """
        Test loading a board cards with all metadata.
        """
        from bs4 import BeautifulSoup

        trello_loader = TrelloLoader.from_credentials(
            "QA",
            api_key="API_KEY",
            token="API_TOKEN",
        )
        documents = trello_loader.load()
        self.assertEqual(len(documents), len(TRELLO_CARDS_QA), "Card count matches.")

        soup = BeautifulSoup(documents[0].page_content, "html.parser")
        self.assertTrue(
            len(soup.find_all()) == 0,
            "There is not markup in Closed Card document content.",
        )

        # Check samples of every field type is present in page content.
        texts = [
            "Closed Card Title",
            "This is the description of Closed Card.",
            "Checklist 1",
            "Item 1:pending",
            "This is a comment on a Closed Card.",
        ]
        for text in texts:
            self.assertTrue(text in documents[0].page_content)

        # Check all metadata is present in first Card
        self.assertEqual(
            documents[0].metadata,
            {
                "title": "Closed Card Title",
                "id": "12350aca6952888df7975903",
                "url": "https://trello.com/card/12350aca6952888df7975903",
                "labels": [],
                "list": "Done",
                "closed": True,
                "due_date": "",
            },
            "Metadata of Closed Card Matches.",
        )

        soup = BeautifulSoup(documents[1].page_content, "html.parser")
        self.assertTrue(
            len(soup.find_all()) == 0,
            "There is not markup in Card 2 document content.",
        )

        # Check samples of every field type is present in page content.
        texts = [
            "Card 2",
            "This is the description of Card 2.",
        ]
        for text in texts:
            self.assertTrue(text in documents[1].page_content)

        # Check all metadata is present in second Card
        self.assertEqual(
            documents[1].metadata,
            {
                "title": "Card 2",
                "id": "45650aca6952888df7975903",
                "url": "https://trello.com/card/45650aca6952888df7975903",
                "labels": ["Medium", "Task"],
                "list": "In Progress",
                "closed": False,
                "due_date": "",
            },
            "Metadata of Card 2 Matches.",
        )

        soup = BeautifulSoup(documents[2].page_content, "html.parser")
        self.assertTrue(
            len(soup.find_all()) == 0,
            "There is not markup in Card 2 document content.",
        )

        # Check samples of every field type is present in page content.
        texts = [
            "Camera",
            "camera limit to stage size:complete",
            "Use 'Impulse' Cinemachine feature for camera shake.",
        ]

        for text in texts:
            self.assertTrue(text in documents[2].page_content, text + " is present.")

        # Check all metadata is present in second Card
        self.assertEqual(
            documents[2].metadata,
            {
                "title": "Camera",
                "id": "55550aca6952888df7975903",
                "url": "https://trello.com/card/55550aca6952888df7975903",
                "labels": ["Task"],
                "list": "Selected for Milestone",
                "closed": False,
                "due_date": "",
            },
            "Metadata of Camera Card matches.",
        )

    def test_partial_text_and_metadata(self) -> None:
        """
        Test loading a board cards removing some text and metadata.
        """
        trello_loader = TrelloLoader.from_credentials(
            "QA",
            api_key="API_KEY",
            token="API_TOKEN",
            extra_metadata=("list"),
            include_card_name=False,
            include_checklist=False,
            include_comments=False,
        )
        documents = trello_loader.load()

        # Check samples of every field type is present in page content.
        texts = [
            "Closed Card Title",
            "Checklist 1",
            "Item 1:pending",
            "This is a comment on a Closed Card.",
        ]
        for text in texts:
            self.assertFalse(text in documents[0].page_content)

        # Check all metadata is present in first Card
        self.assertEqual(
            documents[0].metadata,
            {
                "title": "Closed Card Title",
                "id": "12350aca6952888df7975903",
                "url": "https://trello.com/card/12350aca6952888df7975903",
                "list": "Done",
            },
            "Metadata of Closed Card Matches.",
        )
