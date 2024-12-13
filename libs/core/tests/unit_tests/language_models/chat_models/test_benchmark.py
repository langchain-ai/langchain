import time
from itertools import cycle

from langchain_core.language_models import GenericFakeChatModel


def test_benchmark_model() -> None:
    """Add rate limiter."""
    tic = time.time()

    model = GenericFakeChatModel(
        messages=cycle(["hello", "world", "!"]),
    )

    for _ in range(1_000):
        model.invoke("foo")
    toc = time.time()

    # Verify that the time taken to run the loop is less than 1 seconds

    assert (toc - tic) < 1
