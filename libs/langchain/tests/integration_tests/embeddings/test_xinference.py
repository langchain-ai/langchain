"""Test Xinference embeddings."""
import time
from typing import AsyncGenerator, Tuple

import pytest_asyncio

from langchain.embeddings import XinferenceEmbeddings


@pytest_asyncio.fixture
async def setup() -> AsyncGenerator[Tuple[str, str], None]:
    import xoscar as xo
    from xinference.deploy.supervisor import start_supervisor_components
    from xinference.deploy.utils import create_worker_actor_pool
    from xinference.deploy.worker import start_worker_components

    pool = await create_worker_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}"
    )
    print(f"Pool running on localhost:{pool.external_address}")

    endpoint = await start_supervisor_components(
        pool.external_address, "127.0.0.1", xo.utils.get_next_port()
    )
    await start_worker_components(
        address=pool.external_address, supervisor_address=pool.external_address
    )

    # wait for the api.
    time.sleep(3)
    async with pool:
        yield endpoint, pool.external_address


def test_xinference_embedding_documents(setup: Tuple[str, str]) -> None:
    """Test xinference embeddings for documents."""
    from xinference.client import RESTfulClient

    endpoint, _ = setup

    client = RESTfulClient(endpoint)

    model_uid = client.launch_model(
        model_name="vicuna-v1.3",
        model_size_in_billions=7,
        model_format="ggmlv3",
        quantization="q4_0",
    )

    xinference = XinferenceEmbeddings(server_url=endpoint, model_uid=model_uid)

    documents = ["foo bar", "bar foo"]
    output = xinference.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 4096


def test_xinference_embedding_query(setup: Tuple[str, str]) -> None:
    """Test xinference embeddings for query."""
    from xinference.client import RESTfulClient

    endpoint, _ = setup

    client = RESTfulClient(endpoint)

    model_uid = client.launch_model(
        model_name="vicuna-v1.3", model_size_in_billions=7, quantization="q4_0"
    )

    xinference = XinferenceEmbeddings(server_url=endpoint, model_uid=model_uid)

    document = "foo bar"
    output = xinference.embed_query(document)
    assert len(output) == 4096
