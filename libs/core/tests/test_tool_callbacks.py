from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import Tool


class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    def on_chain_start(self, *args, **kwargs):
        self.events.append("chain_start")

    def on_chain_end(self, *args, **kwargs):
        self.events.append("chain_end")

    def on_tool_start(self, *args, **kwargs):
        self.events.append("tool_start")

    def on_tool_end(self, *args, **kwargs):
        self.events.append("tool_end")


def test_tool_used_as_runnable_emits_chain_and_tool_events():
    def echo(x: str) -> str:
        return x

    tool = Tool.from_function(
        func=echo,
        name="echo",
        description="Echo tool",
    )

    recorder = Recorder()

    config = RunnableConfig(callbacks=[recorder])

    runnable = RunnableLambda(lambda _: tool.run("hello")).with_config(config)

    output = runnable.invoke(None)

    assert output == "hello"

    # This is the BUG: behavior is inconsistent today
    # This is the EXPECTED behavior we want to enforce
    assert recorder.events == [
        "chain_start",
        "tool_start",
        "tool_end",
        "chain_end",
    ]
