from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import torch

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = [f"document {i} about running embeddings on accelerators" for i in range(32)]


def test_embeddings_run_on_accelerator(
    accelerator_device: int | str,
    assert_on_accelerator: Callable[[torch.device], None],
) -> None:
    """Embed on the requested accelerator and confirm placement is not CPU."""
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": str(torch.device(accelerator_device))},
    )

    assert len(embeddings.embed_query("hello")) > 0
    assert_on_accelerator(embeddings._client.device)


def test_multi_process_embeddings_run_on_accelerators(
    accelerator_device: int | str,
) -> None:
    """Start one worker per accelerator and match single-process output."""
    expected = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": str(torch.device(accelerator_device))},
    ).embed_documents(TEXTS)
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, multi_process=True)
    # Spy on the class, not the instance: the instance is pickled into the workers.
    client_cls = type(embeddings._client)

    with patch.object(
        client_cls,
        "start_multi_process_pool",
        autospec=True,
        side_effect=client_cls.start_multi_process_pool,
    ) as start_pool:
        result = embeddings.embed_documents(TEXTS)

    _, target_devices = start_pool.call_args.args
    assert target_devices
    assert all(torch.device(d).type != "cpu" for d in target_devices)
    np.testing.assert_allclose(result, expected, atol=1e-4)
