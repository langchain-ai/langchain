"""Steamship Utils."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from steamship import Block, Steamship


def make_image_public(client: Steamship, block: Block) -> str:
    """Upload a block to a signed URL and return the public URL."""
    try:
        from steamship.data.workspace import SignedUrl
        from steamship.utils.signed_urls import upload_to_signed_url
    except ImportError:
        raise ValueError(
            "The make_image_public function requires the steamship"
            " package to be installed. Please install steamship"
            " with `pip install --upgrade steamship`"
        )

    filepath = str(uuid.uuid4())
    signed_url = (
        client.get_workspace()
        .create_signed_url(
            SignedUrl.Request(
                bucket=SignedUrl.Bucket.PLUGIN_DATA,
                filepath=filepath,
                operation=SignedUrl.Operation.WRITE,
            )
        )
        .signed_url
    )
    read_signed_url = (
        client.get_workspace()
        .create_signed_url(
            SignedUrl.Request(
                bucket=SignedUrl.Bucket.PLUGIN_DATA,
                filepath=filepath,
                operation=SignedUrl.Operation.READ,
            )
        )
        .signed_url
    )
    upload_to_signed_url(signed_url, block.raw())
    return read_signed_url
