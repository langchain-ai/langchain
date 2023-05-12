import pathlib
import os
import time

from pytest_mock import MockerFixture
from langchain.document_loaders import EverNoteLoader


class TestEverNoteLoader:

    @staticmethod
    def example_notebook_path(notebook_name: str) -> str:
        current_dir = pathlib.Path(__file__).parent
        return os.path.join(current_dir, "sample_documents", notebook_name)

    def test_evernoteloader_loadnotebook_eachnoteisindividualdocument(self) -> None:
        loader = EverNoteLoader(self.example_notebook_path("sample_notebook.enex"))
        documents = loader.load()
        assert len(documents) == 2

    def test_evernoteloader_loadnotebook_eachnotehasexpectedcontent(self) -> None:
        documents = EverNoteLoader(self.example_notebook_path("sample_notebook.enex")).load()

        content_note1 = documents[0].page_content
        assert content_note1 == "abc\n\n"

        content_note2 = documents[1].page_content
        assert content_note2 == "**Jan - March 2022**\n\n"

    def test_evernoteloader_loadnotebook_eachnotehasexpectedmetadata(self) -> None:

        documents = EverNoteLoader(self.example_notebook_path("sample_notebook.enex")).load()
        metadata_note1 = documents[0].metadata

        assert "title" in metadata_note1.keys()
        assert "created" in metadata_note1.keys()
        assert "updated" in metadata_note1.keys()
        assert "note-attributes.author" in metadata_note1.keys()

        assert metadata_note1["title"] == "Test"
        assert metadata_note1["note-attributes.author"] == "Michael McGarry"

        assert isinstance(metadata_note1["created"], time.struct_time)
        assert isinstance(metadata_note1["updated"], time.struct_time)

        assert metadata_note1["created"].tm_year == 2023
        assert metadata_note1["created"].tm_mon == 5
        assert metadata_note1["created"].tm_mday == 11

        assert metadata_note1["updated"].tm_year == 2024
        assert metadata_note1["updated"].tm_mon == 7
        assert metadata_note1["updated"].tm_mday == 14

        metadata_note2 = documents[1].metadata

        assert "title" in metadata_note2.keys()
        assert "created" in metadata_note2.keys()
        assert "updated" not in metadata_note2.keys()
        assert "note-attributes.author" in metadata_note2.keys()
        assert "note-attributes.source" in metadata_note2.keys()

        assert metadata_note2["title"] == "Summer Training Program"
        assert metadata_note2["note-attributes.author"] == "Mike McGarry"
        assert metadata_note2["note-attributes.source"] == "mobile.iphone"

        assert isinstance(metadata_note2["created"], time.struct_time)

        assert metadata_note2["created"].tm_year == 2022
        assert metadata_note2["created"].tm_mon == 12
        assert metadata_note2["created"].tm_mday == 27

    def test_evernoteloader_loadnotebookwithconflictingsourcemetadatatag_sourceoffilepreferredoversourceinnotemetadata(self) -> None:
        documents = EverNoteLoader(self.example_notebook_path("sample_notebook_2.enex")).load()
        assert "sample_notebook_2.enex" in documents[0].metadata["source"]
        assert "mobile.iphone" not in documents[0].metadata["source"]
