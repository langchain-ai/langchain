import os
import pathlib
import time

import pytest

from langchain.document_loaders import EverNoteLoader


@pytest.mark.requires("lxml", "html2text")
class TestEverNoteLoader:
    @staticmethod
    def example_notebook_path(notebook_name: str) -> str:
        current_dir = pathlib.Path(__file__).parent
        return os.path.join(current_dir, "sample_documents", notebook_name)

    def test_loadnotebook_eachnoteisindividualdocument(self) -> None:
        loader = EverNoteLoader(
            self.example_notebook_path("sample_notebook.enex"), False
        )
        documents = loader.load()
        assert len(documents) == 2

    def test_loadnotebook_eachnotehasexpectedcontentwithleadingandtrailingremoved(
        self,
    ) -> None:
        documents = EverNoteLoader(
            self.example_notebook_path("sample_notebook.enex"), False
        ).load()

        content_note1 = documents[0].page_content
        assert content_note1 == "abc"

        content_note2 = documents[1].page_content
        assert content_note2 == "**Jan - March 2022**"

    def test_loademptynotebook_emptylistreturned(self) -> None:
        documents = EverNoteLoader(
            self.example_notebook_path("empty_export.enex"), False
        ).load()
        assert len(documents) == 0

    def test_loadnotewithemptycontent_emptydocumentcontent(self) -> None:
        documents = EverNoteLoader(
            self.example_notebook_path("sample_notebook_emptynote.enex"), False
        ).load()
        note = documents[0]
        assert note.page_content == ""

    def test_loadnotewithmissingcontenttag_emptylistreturned(
        self,
    ) -> None:
        documents = EverNoteLoader(
            self.example_notebook_path("sample_notebook_missingcontenttag.enex"), False
        ).load()
        assert len(documents) == 0

    def test_loadnotewithnometadata_documentreturnedwithsourceonly(
        self,
    ) -> None:
        documents = EverNoteLoader(
            self.example_notebook_path("sample_notebook_missingmetadata.enex"), False
        ).load()
        note = documents[0]

        assert note.page_content == "I only have content, no metadata"

        assert len(note.metadata) == 1
        assert "source" in note.metadata
        assert "sample_notebook_missingmetadata.enex" in note.metadata["source"]

    def test_loadnotebookwithimage_notehasplaintextonlywithresourcesremoved(
        self,
    ) -> None:
        documents = EverNoteLoader(
            self.example_notebook_path("sample_notebook_with_media.enex"), False
        ).load()

        note = documents[0]
        assert (
            note.page_content
            == """\
When you pick this mug up with your thumb on top and middle finger through the
loop, your ring finger slides into the mug under the loop where it is too hot
to touch and burns you.

  

If you try and pick it up with your thumb and index finger you can’t hold the
mug."""
        )

    def test_loadnotebook_eachnotehasexpectedmetadata(self) -> None:
        documents = EverNoteLoader(
            self.example_notebook_path("sample_notebook.enex"), False
        ).load()
        metadata_note1 = documents[0].metadata

        assert "title" in metadata_note1.keys()
        assert "created" in metadata_note1.keys()
        assert "updated" in metadata_note1.keys()
        assert "note-attributes.author" in metadata_note1.keys()
        assert (
            "content" not in metadata_note1.keys()
        )  # This should be in the content of the document instead
        assert (
            "content-raw" not in metadata_note1.keys()
        )  # This is too large to be stored as metadata
        assert (
            "resource" not in metadata_note1.keys()
        )  # This is too large to be stored as metadata

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
        assert "content" not in metadata_note2.keys()
        assert "content-raw" not in metadata_note2.keys()
        assert (
            "resource" not in metadata_note2.keys()
        )  # This is too large to be stored as metadata

        assert metadata_note2["title"] == "Summer Training Program"
        assert metadata_note2["note-attributes.author"] == "Mike McGarry"
        assert metadata_note2["note-attributes.source"] == "mobile.iphone"

        assert isinstance(metadata_note2["created"], time.struct_time)

        assert metadata_note2["created"].tm_year == 2022
        assert metadata_note2["created"].tm_mon == 12
        assert metadata_note2["created"].tm_mday == 27

    def test_loadnotebookwithconflictingsourcemetadatatag_sourceoffilepreferred(
        self,
    ) -> None:
        documents = EverNoteLoader(
            self.example_notebook_path("sample_notebook_2.enex"), False
        ).load()
        assert "sample_notebook_2.enex" in documents[0].metadata["source"]
        assert "mobile.iphone" not in documents[0].metadata["source"]

    def test_returnsingledocument_loadnotebook_eachnoteiscombinedinto1document(
        self,
    ) -> None:
        loader = EverNoteLoader(
            self.example_notebook_path("sample_notebook.enex"), True
        )
        documents = loader.load()
        assert len(documents) == 1

    def test_returnsingledocument_loadnotebook_notecontentiscombinedinto1document(
        self,
    ) -> None:
        loader = EverNoteLoader(
            self.example_notebook_path("sample_notebook.enex"), True
        )
        documents = loader.load()
        note = documents[0]
        assert note.page_content == "abc**Jan - March 2022**"
