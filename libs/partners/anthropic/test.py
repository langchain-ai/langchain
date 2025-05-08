import cProfile
import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk


MODEL_NAME = "claude-3-5-haiku-latest"


async def test_stream_usage() -> None:
    """Test usage metadata can be excluded."""
    model = ChatAnthropic(model_name=MODEL_NAME, stream_usage=False)
    async for token in model.astream("hi"):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None
    # check we override with kwarg
    model = ChatAnthropic(model_name=MODEL_NAME)
    assert model.stream_usage
    async for token in model.astream("hi", stream_usage=False):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None


# def run_async(func, *args, **kwargs):
#     return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     run_async(test_stream_usage)   # your async function here
#     profiler.disable()
#     profiler.dump_stats('test_stream_usage_master.prof')  # or 'test_stream_usage.out'



import yappi


def run_async(func, *args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))

yappi.start()
run_async(test_stream_usage)
yappi.stop()
yappi.get_func_stats().save('profile.yappi', type='pstat')  # pstat format
