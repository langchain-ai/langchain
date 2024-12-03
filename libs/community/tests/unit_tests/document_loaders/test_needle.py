import pytest
from pytest_mock import MockerFixture


@pytest.mark.requires("needle")
def test_add_and_fetch_files(mocker: MockerFixture) -> None:
    """
    Test adding and fetching files using the NeedleLoader with a mock.
    """
    from langchain_community.document_loaders.needle import NeedleLoader  # noqa: I001
    from needle.v1.models import CollectionFile  # noqa: I001

    # Create mock instances using mocker
    # Create mock instances using mocker
    mock_files = mocker.Mock()
    mock_files.add.return_value = [
        CollectionFile(
            id="mock_id",
            name="tech-radar-30.pdf",
            url="https://example.com/",
            status="indexed",
            type="mock_type",
            user_id="mock_user_id",
            connector_id="mock_connector_id",
            size=1234,
            md5_hash="mock_md5_hash",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
    ]
    mock_files.list.return_value = [
        CollectionFile(
            id="mock_id",
            name="tech-radar-30.pdf",
            url="https://example.com/",
            status="indexed",
            type="mock_type",
            user_id="mock_user_id",
            connector_id="mock_connector_id",
            size=1234,
            md5_hash="mock_md5_hash",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
    ]

    mock_collections = mocker.Mock()
    mock_collections.files = mock_files

    mock_needle_client = mocker.Mock()
    mock_needle_client.collections = mock_collections

    # Patch the NeedleClient to return the mock client
    mocker.patch("needle.v1.NeedleClient", return_value=mock_needle_client)

    # Initialize NeedleLoader with mock API key and collection ID
    document_store = NeedleLoader(
        needle_api_key="fake_api_key",
        collection_id="fake_collection_id",
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
    assert isinstance(added_files[0].metadata["title"], str)
    assert isinstance(added_files[0].metadata["source"], str)
