import base64
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents.base import Blob
from mcp.types import (
    BlobResourceContents,
    ListResourcesResult,
    ReadResourceResult,
    Resource,
    ResourceContents,
    TextResourceContents,
)

from langchain.mcp.resources import (
    convert_mcp_resource_to_langchain_blob,
    get_mcp_resource,
    load_mcp_resources,
)


def test_convert_mcp_resource_to_langchain_blob_with_text():
    uri = "file:///test.txt"
    contents = TextResourceContents(uri=uri, mimeType="text/plain", text="Hello, world!")

    blob = convert_mcp_resource_to_langchain_blob(uri, contents)

    assert isinstance(blob, Blob)
    assert blob.data == "Hello, world!"
    assert blob.mimetype == "text/plain"
    assert blob.metadata["uri"] == uri


def test_convert_mcp_resource_to_langchain_blob():
    uri = "file:///test.png"
    original_data = b"binary-image-data"
    base64_blob = base64.b64encode(original_data).decode()

    contents = BlobResourceContents(uri=uri, mimeType="image/png", blob=base64_blob)

    blob = convert_mcp_resource_to_langchain_blob(uri, contents)

    assert isinstance(blob, Blob)
    assert blob.data == original_data
    assert blob.mimetype == "image/png"
    assert blob.metadata["uri"] == uri


def test_convert_mcp_resource_to_langchain_blob_with_invalid_type():
    class DummyContent(ResourceContents):
        pass

    with pytest.raises(ValueError):
        convert_mcp_resource_to_langchain_blob("file:///dummy", DummyContent())


async def test_get_mcp_resource_with_contents():
    session = AsyncMock()
    uri = "file:///test.txt"

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="text/plain", text="Content 1"),
                TextResourceContents(uri=uri, mimeType="text/plain", text="Content 2"),
            ],
        ),
    )

    blobs = await get_mcp_resource(session, uri)

    assert len(blobs) == 2
    assert all(isinstance(d, Blob) for d in blobs)
    assert blobs[0].data == "Content 1"
    assert blobs[1].data == "Content 2"


async def test_get_mcp_resource_with_text_and_blob():
    session = AsyncMock()
    uri = "file:///mixed"

    original_data = b"some-binary-content"
    base64_blob = base64.b64encode(original_data).decode()

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="text/plain", text="Hello Text"),
                BlobResourceContents(
                    uri=uri,
                    mimeType="application/octet-stream",
                    blob=base64_blob,
                ),
            ],
        ),
    )

    results = await get_mcp_resource(session, uri)

    assert len(results) == 2

    assert isinstance(results[0], Blob)
    assert results[0].data == "Hello Text"
    assert results[0].mimetype == "text/plain"

    assert isinstance(results[1], Blob)
    assert results[1].data == original_data
    assert results[1].mimetype == "application/octet-stream"


async def test_get_mcp_resource_with_empty_contents():
    session = AsyncMock()
    uri = "file:///empty.txt"

    session.read_resource = AsyncMock(return_value=ReadResourceResult(contents=[]))

    blobs = await get_mcp_resource(session, uri)

    assert len(blobs) == 0
    session.read_resource.assert_called_once_with(uri)


async def test_load_mcp_resources_with_list_of_uris():
    session = AsyncMock()
    uri1 = "file:///test1.txt"
    uri2 = "file:///test2.txt"

    session.read_resource = AsyncMock()
    session.read_resource.side_effect = [
        ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri1, mimeType="text/plain", text="Content from test1"),
            ],
        ),
        ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri2, mimeType="text/plain", text="Content from test2"),
            ],
        ),
    ]

    blobs = await load_mcp_resources(session, uris=[uri1, uri2])

    assert len(blobs) == 2
    assert all(isinstance(d, Blob) for d in blobs)
    assert blobs[0].data == "Content from test1"
    assert blobs[1].data == "Content from test2"
    assert blobs[0].metadata["uri"] == uri1
    assert blobs[1].metadata["uri"] == uri2
    assert session.read_resource.call_count == 2


async def test_load_mcp_resources_with_single_uri_string():
    session = AsyncMock()
    uri = "file:///test.txt"

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(
            contents=[
                TextResourceContents(uri=uri, mimeType="text/plain", text="Content from test"),
            ],
        ),
    )

    blobs = await load_mcp_resources(session, uris=uri)

    assert len(blobs) == 1
    assert isinstance(blobs[0], Blob)
    assert blobs[0].data == "Content from test"
    assert blobs[0].metadata["uri"] == uri
    session.read_resource.assert_called_once_with(uri)


async def test_load_mcp_resources_with_all_resources():
    session = AsyncMock()

    session.list_resources = AsyncMock(
        return_value=ListResourcesResult(
            resources=[
                Resource(uri="file:///test1.txt", name="test1.txt", mimeType="text/plain"),
                Resource(uri="file:///test2.txt", name="test2.txt", mimeType="text/plain"),
            ],
        ),
    )

    session.read_resource = AsyncMock()
    session.read_resource.side_effect = [
        ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri="file:///test1.txt",
                    mimeType="text/plain",
                    text="Content from test1",
                ),
            ],
        ),
        ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri="file:///test2.txt",
                    mimeType="text/plain",
                    text="Content from test2",
                ),
            ],
        ),
    ]

    blobs = await load_mcp_resources(session)

    assert len(blobs) == 2
    assert blobs[0].data == "Content from test1"
    assert blobs[1].data == "Content from test2"
    assert session.list_resources.called
    assert session.read_resource.call_count == 2


async def test_load_mcp_resources_with_error_handling():
    session = AsyncMock()
    uri1 = "file:///valid.txt"
    uri2 = "file:///error.txt"

    session.read_resource = AsyncMock()
    session.read_resource.side_effect = [
        ReadResourceResult(
            contents=[TextResourceContents(uri=uri1, mimeType="text/plain", text="Valid content")],
        ),
        Exception("Resource not found"),
    ]

    with pytest.raises(RuntimeError) as exc_info:
        await load_mcp_resources(session, uris=[uri1, uri2])

    assert "Error fetching resource" in str(exc_info.value)


async def test_load_mcp_resources_with_blob_content():
    session = AsyncMock()
    uri = "file:///with_blob"
    original_data = b"binary data"
    base64_blob = base64.b64encode(original_data).decode()

    session.read_resource = AsyncMock(
        return_value=ReadResourceResult(
            contents=[
                BlobResourceContents(
                    uri=uri,
                    mimeType="application/octet-stream",
                    blob=base64_blob,
                ),
            ],
        ),
    )

    blobs = await load_mcp_resources(session, uris=uri)

    assert len(blobs) == 1
    assert isinstance(blobs[0], Blob)
    assert blobs[0].data == original_data
    assert blobs[0].mimetype == "application/octet-stream"
