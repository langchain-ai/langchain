import pytest
from needle.v1.models import CollectionFile, FileToAdd
from pytest_mock import MockerFixture

from langchain_community.document_loaders.needle import NeedleLoader


# Mock NeedleClient class to simulate the Needle API interaction
class MockNeedleClient:
    def __init__(self):
        self.collections = MockCollections()


class MockCollections:
    def __init__(self):
        self.files = MockFiles()


class MockFiles:
    def add(self, collection_id: str, files: list[FileToAdd]):
        # Simulate adding files successfully
        return [
            CollectionFile(id="mock_id", name=file.name, url=file.url, status="indexed")
            for file in files
        ]

    def list(self, collection_id: str):
        # Simulate listing files from the collection
        return [
            CollectionFile(
                id="mock_id",
                name="tech-radar-30.pdf",
                url="https://example.com/",
                status="indexed",
            )
        ]


# Need to pass real API key and collection ID to test this function, otherwise fails
@pytest.mark.usefixtures("socket_enabled")
def test_add_and_fetch_files(mocker: MockerFixture):
    # Mock the NeedleClient to use the mock implementation
    mocker.patch("needle.v1.NeedleClient", new=MockNeedleClient)

    # Initialize NeedleLoader with mock API key and collection ID
    document_store = NeedleLoader(
        needle_api_key="apk_01JCTG9FR5ASYNRV0RJSD5V6WA",
        collection_id="clt_01JCTG92CX08SSDCPP0P7EQCGR",
    )

    # Define files to add
    files = {
        "tech-radar-30.pdf": "https://www.thoughtworks.com/content/dam/thoughtworks/documents/radar/2024/04/tr_technology_radar_vol_30_en.pdf"
    }

    # Add files to the collection using the mock client
    document_store.add_files(files=files)

    # Fetch the added files using the mock client
    added_files = document_store._fetch_documents()

    # Assertions to verify that the file was added and fetched correctly
    assert type(added_files[0].metadata["title"]) == str
    assert type(added_files[0].metadata["source"]) == str

    print("Test passed: Files added and fetched successfully.")
