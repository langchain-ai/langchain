import pytest
from langchain_needle import NeedleLoader
from pytest_mock import MockerFixture
from needle.v1.models import FileToAdd, CollectionFile

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
        return [CollectionFile(id="mock_id", name=file.name, url=file.url, status="indexed") for file in files]

    def list(self, collection_id: str):
        # Simulate listing files from the collection
        return [
            CollectionFile(id="mock_id", name="tech-radar-30.pdf", url="https://mock-url.com", status="indexed")
        ]

# Need to pass real API key and collection ID to test this function, otherwise fails
@pytest.mark.usefixtures("socket_enabled")
def test_add_and_fetch_files(mocker: MockerFixture):
    # Mock the NeedleClient to use the mock implementation
    mocker.patch("needle.v1.NeedleClient", new=MockNeedleClient)

    # Initialize NeedleLoader with mock API key and collection ID
    document_store = NeedleLoader(
        needle_api_key="YOUR_API_KEY",
        collection_id="YOUR_COLLECTION_ID"
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
    assert added_files[0].metadata["title"] == "tech-radar-30.pdf"
    assert added_files[0].metadata["source"] == "https://mock-url.com"
    assert added_files[0].page_content == ""  # Mocked empty content

    print("Test passed: Files added and fetched successfully.")

