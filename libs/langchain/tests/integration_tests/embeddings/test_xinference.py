"""Test Xinference embeddings."""
from langchain.embeddings import XinferenceEmbeddings

import time

import pytest_asyncio

@pytest_asyncio.fixture
async def setup():
    try:
        import xinference
        import xoscar as xo
    except ImportError as e:
        raise ImportError(
            "Could not import xinference or xoscar. Make sure to install them in advance"
        ) from e

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

def test_xinference_embedding_documents(setup) -> None:
    """Test xinference embeddings for documents."""
    try:
        from xinference.client import RESTfulClient
    except ImportError as e:
        raise ImportError(
            "Could not import RESTfulClient from xinference. Make sure to install xinference in advance"
        ) from e

    endpoint, _ = setup

    client = RESTfulClient(endpoint)

    model_uid = client.launch_model(
        model_name="orca", model_size_in_billions=3, quantization="q4_0", embedding="True"
    )

    xinference = XinferenceEmbeddings(
        server_url=endpoint,
        model_uid=model_uid
    )

    documents = ["foo bar", "bar foo"]
    output = xinference.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 3200


def test_xinference_embedding_query(setup) -> None:
    """Test xinference embeddings for query."""
    try:
        from xinference.client import RESTfulClient
    except ImportError as e:
        raise ImportError(
            "Could not import RESTfulClient from xinference. Make sure to install xinference in advance"
        ) from e

    endpoint, _ = setup

    client = RESTfulClient(endpoint)

    model_uid = client.launch_model(
        model_name="orca", model_size_in_billions=3, quantization="q4_0", embedding="True"
    )

    xinference = XinferenceEmbeddings(
        server_url=endpoint,
        model_uid=model_uid
    )

    document = "foo bar"
    output = xinference.embed_query(document)
    assert len(output) == 3200