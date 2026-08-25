"""Resources adapter for converting MCP resources to LangChain [Blob objects][langchain_core.documents.base.Blob].

This module provides functionality to convert MCP resources into LangChain Blob
objects, handling both text and binary resource content types.
"""  # noqa: E501

import base64

from langchain_core.documents.base import Blob
from mcp import ClientSession
from mcp.types import BlobResourceContents, ResourceContents, TextResourceContents


def convert_mcp_resource_to_langchain_blob(resource_uri: str, contents: ResourceContents) -> Blob:
    """Convert an MCP resource content to a LangChain Blob.

    Args:
        resource_uri: URI of the resource
        contents: The resource contents

    Returns:
        A LangChain Blob

    """
    data: str | bytes
    if isinstance(contents, TextResourceContents):
        data = contents.text
    elif isinstance(contents, BlobResourceContents):
        data = base64.b64decode(contents.blob)
    else:
        msg = f"Unsupported content type for URI {resource_uri}"
        raise TypeError(msg)

    return Blob.from_data(data=data, mime_type=contents.mime_type, metadata={"uri": resource_uri})


async def get_mcp_resource(session: ClientSession, uri: str) -> list[Blob]:
    """Fetch a single MCP resource and convert it to LangChain [Blob objects][langchain_core.documents.base.Blob].

    Args:
        session: MCP client session.
        uri: URI of the resource to fetch.

    Returns:
        A list of LangChain [Blob][langchain_core.documents.base.Blob] objects.
    """  # noqa: E501
    contents_result = await session.read_resource(uri)
    if not contents_result.contents or len(contents_result.contents) == 0:
        return []

    return [
        convert_mcp_resource_to_langchain_blob(uri, content) for content in contents_result.contents
    ]


async def load_mcp_resources(
    session: ClientSession,
    *,
    uris: str | list[str] | None = None,
) -> list[Blob]:
    """Load MCP resources and convert them to LangChain [Blob objects][langchain_core.documents.base.Blob].

    Args:
        session: MCP client session.
        uris: List of URIs to load. If `None`, all resources will be loaded.

            !!! note

                Dynamic resources will NOT be loaded when `None` is specified,
                as they require parameters and are ignored by the MCP SDK's
                `session.list_resources()` method.

    Returns:
        A list of LangChain [Blob][langchain_core.documents.base.Blob] objects.

    Raises:
        RuntimeError: If an error occurs while fetching a resource.
    """  # noqa: E501
    blobs = []

    if uris is None:
        resources_list = await session.list_resources()
        uri_list = [r.uri for r in resources_list.resources]
    elif isinstance(uris, str):
        uri_list = [uris]
    else:
        uri_list = uris

    current_uri = None
    try:
        for uri in uri_list:
            current_uri = uri
            resource_blobs = await get_mcp_resource(session, uri)
            blobs.extend(resource_blobs)
    except Exception as e:
        msg = f"Error fetching resource {current_uri}"
        raise RuntimeError(msg) from e

    return blobs
