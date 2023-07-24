"""Test Xinference wrapper."""
from langchain.llms import Xinference

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


def test_xinference_llm_(setup) -> None:
    try:
        from xinference.client import RESTfulClient
    except ImportError as e:
        raise ImportError(
            "Could not import RESTfulClient from xinference. Make sure to install xinference in advance"
        ) from e

    endpoint, _ = setup

    client = RESTfulClient(endpoint)

    model_uid = client.launch_model(
        model_name="vicuna-v1.3", model_size_in_billions=7, quantization="q4_0"
    )

    llm = Xinference(server_url=endpoint, model_uid=model_uid)

    answer = llm(prompt="Q: What food can we try in the capital of France? A:")

    assert isinstance(answer, str)

    answer = llm(
        prompt="Q: where can we visit in the capital of France? A:",
        generate_config={"max_tokens": 1024, "stream": True},
    )

    assert isinstance(answer, str)
