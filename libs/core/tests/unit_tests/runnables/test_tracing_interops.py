from __future__ import annotations

import json
import sys
import uuid
from inspect import isasyncgenfunction
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import MagicMock, patch

import pytest
from langsmith import Client, get_current_run_tree, traceable
from langsmith.run_helpers import tracing_context
from langsmith.utils import get_env_var

from langchain_core.runnables.base import RunnableLambda, RunnableParallel
from langchain_core.tracers.langchain import LangChainTracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine, Generator

    from langsmith.run_trees import RunTree

    from langchain_core.callbacks import BaseCallbackHandler


def _get_posts(client: Client) -> list:
    mock_calls = client.session.request.mock_calls  # type: ignore[attr-defined]
    posts = []
    for call in mock_calls:
        if call.args:
            if call.args[0] != "POST":
                continue
            assert call.args[0] == "POST"
            assert call.args[1].startswith("https://api.smith.langchain.com")
            body = json.loads(call.kwargs["data"])
            if "post" in body:
                # Batch request
                assert body["post"]
                posts.extend(body["post"])
            else:
                posts.append(body)
    return posts


def _create_tracer_with_mocked_client(
    project_name: str | None = None,
    tags: list[str] | None = None,
) -> LangChainTracer:
    mock_session = MagicMock()
    mock_client_ = Client(
        session=mock_session, api_key="test", auto_batch_tracing=False
    )
    return LangChainTracer(client=mock_client_, project_name=project_name, tags=tags)


def test_tracing_context() -> None:
    mock_session = MagicMock()
    mock_client_ = Client(
        session=mock_session, api_key="test", auto_batch_tracing=False
    )

    @RunnableLambda
    def my_lambda(a: int) -> int:
        return a + 1

    @RunnableLambda
    def my_function(a: int) -> int:
        with tracing_context(enabled=False):
            return my_lambda.invoke(a)

    name = uuid.uuid4().hex
    project_name = f"Some project {name}"
    with tracing_context(project_name=project_name, client=mock_client_, enabled=True):
        assert my_function.invoke(1) == 2
    posts = _get_posts(mock_client_)
    assert len(posts) == 1
    assert all(post["session_name"] == project_name for post in posts)


def test_config_traceable_handoff() -> None:
    get_env_var.cache_clear()
    tracer = _create_tracer_with_mocked_client(
        project_name="another-flippin-project", tags=["such-a-tag"]
    )

    @traceable
    def my_great_great_grandchild_function(a: int) -> int:
        rt = get_current_run_tree()
        assert rt
        assert rt.session_name == "another-flippin-project"
        return a + 1

    @RunnableLambda
    def my_great_grandchild_function(a: int) -> int:
        return my_great_great_grandchild_function(a)

    @RunnableLambda
    def my_grandchild_function(a: int) -> int:
        return my_great_grandchild_function.invoke(a)

    @traceable
    def my_child_function(a: int) -> int:
        return my_grandchild_function.invoke(a) * 3

    @traceable()
    def my_function(a: int) -> int:
        rt = get_current_run_tree()
        assert rt
        assert rt.session_name == "another-flippin-project"
        return my_child_function(a)

    def my_parent_function(a: int) -> int:
        rt = get_current_run_tree()
        assert rt
        assert rt.session_name == "another-flippin-project"
        return my_function(a)

    my_parent_runnable = RunnableLambda(my_parent_function)

    assert my_parent_runnable.invoke(1, {"callbacks": [tracer]}) == 6
    posts = _get_posts(tracer.client)
    assert all(post["session_name"] == "another-flippin-project" for post in posts)
    # There should have been 6 runs created,
    # one for each function invocation
    assert len(posts) == 6
    name_to_body = {post["name"]: post for post in posts}

    ordered_names = [
        "my_parent_function",
        "my_function",
        "my_child_function",
        "my_grandchild_function",
        "my_great_grandchild_function",
        "my_great_great_grandchild_function",
    ]
    trace_id = posts[0]["trace_id"]
    last_dotted_order = None
    parent_run_id = None
    for name in ordered_names:
        id_ = name_to_body[name]["id"]
        parent_run_id_ = name_to_body[name].get("parent_run_id")
        if parent_run_id_ is not None:
            assert parent_run_id == parent_run_id_
        assert name in name_to_body
        # All within the same trace
        assert name_to_body[name]["trace_id"] == trace_id
        dotted_order: str = name_to_body[name]["dotted_order"]
        assert dotted_order is not None
        if last_dotted_order is not None:
            assert dotted_order > last_dotted_order
            assert dotted_order.startswith(last_dotted_order), (
                "Unexpected dotted order for run"
                f" {name}\n{dotted_order}\n{last_dotted_order}"
            )
        last_dotted_order = dotted_order
        parent_run_id = id_
    assert "such-a-tag" in name_to_body["my_parent_function"]["tags"]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Asyncio context vars require Python 3.11+"
)
async def test_config_traceable_async_handoff() -> None:
    tracer = _create_tracer_with_mocked_client()

    @traceable
    def my_great_great_grandchild_function(a: int) -> int:
        return a + 1

    @RunnableLambda
    def my_great_grandchild_function(a: int) -> int:
        return my_great_great_grandchild_function(a)

    @RunnableLambda  # type: ignore[arg-type]
    async def my_grandchild_function(a: int) -> int:
        return my_great_grandchild_function.invoke(a)

    @traceable
    async def my_child_function(a: int) -> int:
        return await my_grandchild_function.ainvoke(a) * 3  # type: ignore[arg-type]

    @traceable()
    async def my_function(a: int) -> int:
        return await my_child_function(a)

    async def my_parent_function(a: int) -> int:
        return await my_function(a)

    my_parent_runnable = RunnableLambda(my_parent_function)  # type: ignore[arg-type,var-annotated]
    result = await my_parent_runnable.ainvoke(1, {"callbacks": [tracer]})
    assert result == 6
    posts = _get_posts(tracer.client)
    # There should have been 6 runs created,
    # one for each function invocation
    assert len(posts) == 6
    name_to_body = {post["name"]: post for post in posts}
    ordered_names = [
        "my_parent_function",
        "my_function",
        "my_child_function",
        "my_grandchild_function",
        "my_great_grandchild_function",
        "my_great_great_grandchild_function",
    ]
    trace_id = posts[0]["trace_id"]
    last_dotted_order = None
    parent_run_id = None
    for name in ordered_names:
        id_ = name_to_body[name]["id"]
        parent_run_id_ = name_to_body[name].get("parent_run_id")
        if parent_run_id_ is not None:
            assert parent_run_id == parent_run_id_
        assert name in name_to_body
        # All within the same trace
        assert name_to_body[name]["trace_id"] == trace_id
        dotted_order: str = name_to_body[name]["dotted_order"]
        assert dotted_order is not None
        if last_dotted_order is not None:
            assert dotted_order > last_dotted_order
            assert dotted_order.startswith(last_dotted_order), (
                "Unexpected dotted order for run"
                f" {name}\n{dotted_order}\n{last_dotted_order}"
            )
        last_dotted_order = dotted_order
        parent_run_id = id_


@patch("langchain_core.tracers.langchain.get_client")
@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("env", ["", "true"])
def test_tracing_enable_disable(
    mock_get_client: MagicMock, *, enabled: bool | None, env: str
) -> None:
    mock_session = MagicMock()
    mock_client_ = Client(
        session=mock_session, api_key="test", auto_batch_tracing=False
    )
    mock_get_client.return_value = mock_client_

    def my_func(a: int) -> int:
        return a + 1

    get_env_var.cache_clear()
    env_on = env == "true"
    with (
        patch.dict("os.environ", {"LANGSMITH_TRACING": env}),
        tracing_context(enabled=enabled),
    ):
        RunnableLambda(my_func).invoke(1)

    mock_posts = _get_posts(mock_client_)
    if enabled is True:
        assert len(mock_posts) == 1
    elif enabled is False:
        assert not mock_posts
    elif env_on:
        assert len(mock_posts) == 1
    else:
        assert not mock_posts


class TestRunnableSequenceParallelTraceNesting:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.tracer = _create_tracer_with_mocked_client()

    @staticmethod
    def _create_parent(
        other_thing: Callable[
            [int], Generator[int, None, None] | AsyncGenerator[int, None]
        ],
    ) -> RunnableLambda:
        @RunnableLambda
        def my_child_function(a: int) -> int:
            return a + 2

        parallel = RunnableParallel(
            chain_result=my_child_function.with_config(tags=["atag"]),
            other_thing=other_thing,
        )

        def before(x: int) -> int:
            return x

        def after(x: dict) -> int:
            return x["chain_result"]

        sequence = before | parallel | after
        if isasyncgenfunction(other_thing):

            @RunnableLambda  # type: ignore[arg-type]
            async def parent(a: int) -> int:
                return await sequence.ainvoke(a)

        else:

            @RunnableLambda
            def parent(a: int) -> int:
                return sequence.invoke(a)

        return parent

    def _check_posts(self) -> None:
        posts = _get_posts(self.tracer.client)
        name_order = [
            "parent",
            "RunnableSequence",
            "before",
            "RunnableParallel<chain_result,other_thing>",
            ["my_child_function", "other_thing"],
            "after",
        ]
        expected_parents = {
            "parent": None,
            "RunnableSequence": "parent",
            "before": "RunnableSequence",
            "RunnableParallel<chain_result,other_thing>": "RunnableSequence",
            "my_child_function": "RunnableParallel<chain_result,other_thing>",
            "other_thing": "RunnableParallel<chain_result,other_thing>",
            "after": "RunnableSequence",
        }
        assert len(posts) == sum(
            1 if isinstance(n, str) else len(n) for n in name_order
        )
        prev_dotted_order = None
        dotted_order_map = {}
        id_map = {}
        parent_id_map = {}
        i = 0
        for name in name_order:
            if isinstance(name, list):
                for n in name:
                    matching_post = next(
                        p for p in posts[i : i + len(name)] if p["name"] == n
                    )
                    assert matching_post
                    dotted_order = matching_post["dotted_order"]
                    if prev_dotted_order is not None:
                        assert dotted_order > prev_dotted_order
                    dotted_order_map[n] = dotted_order
                    id_map[n] = matching_post["id"]
                    parent_id_map[n] = matching_post.get("parent_run_id")
                i += len(name)
                continue
            assert posts[i]["name"] == name
            dotted_order = posts[i]["dotted_order"]
            if prev_dotted_order is not None and not str(
                expected_parents[name]  # type: ignore[index]
            ).startswith("RunnableParallel"):
                assert dotted_order > prev_dotted_order, (
                    f"{name} not after {name_order[i - 1]}"
                )
            prev_dotted_order = dotted_order
            if name in dotted_order_map:
                msg = f"Duplicate name {name}"
                raise ValueError(msg)
            dotted_order_map[name] = dotted_order
            id_map[name] = posts[i]["id"]
            parent_id_map[name] = posts[i].get("parent_run_id")
            i += 1

        # Now check the dotted orders
        for name, parent_ in expected_parents.items():
            dotted_order = dotted_order_map[name]
            if parent_ is not None:
                parent_dotted_order = dotted_order_map[parent_]
                assert dotted_order.startswith(parent_dotted_order), (
                    f"{name}, {parent_dotted_order} not in {dotted_order}"
                )
                assert str(parent_id_map[name]) == str(id_map[parent_])
            else:
                assert dotted_order.split(".")[0] == dotted_order

    @pytest.mark.parametrize(
        "method",
        [
            lambda parent, cb: parent.invoke(1, {"callbacks": cb}),
            lambda parent, cb: list(parent.stream(1, {"callbacks": cb}))[-1],
            lambda parent, cb: parent.batch([1], {"callbacks": cb})[0],
        ],
        ids=["invoke", "stream", "batch"],
    )
    def test_sync(
        self, method: Callable[[RunnableLambda, list[BaseCallbackHandler]], int]
    ) -> None:
        def other_thing(_: int) -> Generator[int, None, None]:
            yield 1

        parent = self._create_parent(other_thing)

        # Now run the chain and check the resulting posts
        assert method(parent, [self.tracer]) == 3

        self._check_posts()

    @staticmethod
    async def ainvoke(parent: RunnableLambda, cb: list[BaseCallbackHandler]) -> int:
        return await parent.ainvoke(1, {"callbacks": cb})

    @staticmethod
    async def astream(parent: RunnableLambda, cb: list[BaseCallbackHandler]) -> int:
        return [res async for res in parent.astream(1, {"callbacks": cb})][-1]

    @staticmethod
    async def abatch(parent: RunnableLambda, cb: list[BaseCallbackHandler]) -> int:
        return (await parent.abatch([1], {"callbacks": cb}))[0]

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Asyncio context vars require Python 3.11+"
    )
    @pytest.mark.parametrize("method", [ainvoke, astream, abatch])
    async def test_async(
        self,
        method: Callable[
            [RunnableLambda, list[BaseCallbackHandler]], Coroutine[Any, Any, int]
        ],
    ) -> None:
        async def other_thing(_: int) -> AsyncGenerator[int, None]:
            yield 1

        parent = self._create_parent(other_thing)

        # Now run the chain and check the resulting posts
        assert await method(parent, [self.tracer]) == 3

        self._check_posts()


@pytest.mark.parametrize("parent_type", ["ls", "lc"])
def test_tree_is_constructed(parent_type: Literal["ls", "lc"]) -> None:
    mock_session = MagicMock()
    mock_client_ = Client(
        session=mock_session, api_key="test", auto_batch_tracing=False
    )

    @traceable
    def kitten(x: str) -> str:
        return x

    @RunnableLambda
    def grandchild(x: str) -> str:
        return kitten(x)

    @RunnableLambda
    def child(x: str) -> str:
        return grandchild.invoke(x)

    rid = uuid.uuid4()
    with tracing_context(
        client=mock_client_,
        enabled=True,
        metadata={"some_foo": "some_bar"},
        tags=["afoo"],
    ):
        collected: dict[str, RunTree] = {}

        def collect_run(run: RunTree) -> None:
            collected[str(run.id)] = run

        if parent_type == "ls":

            @traceable
            def parent() -> str:
                return child.invoke("foo")

            assert (
                parent(langsmith_extra={"on_end": collect_run, "run_id": rid}) == "foo"
            )
            assert collected

        else:

            @RunnableLambda
            def parent(_: Any) -> str:
                return child.invoke("foo")

            tracer = LangChainTracer()
            tracer._persist_run = collect_run  # type: ignore[method-assign]

            assert parent.invoke(..., {"run_id": rid, "callbacks": [tracer]}) == "foo"  # type: ignore[attr-defined]
    run = collected.get(str(rid))

    assert run is not None
    assert run.name == "parent"
    assert run.child_runs
    child_run = run.child_runs[0]
    assert child_run.name == "child"
    assert child_run.child_runs
    grandchild_run = child_run.child_runs[0]
    assert grandchild_run.name == "grandchild"
    assert grandchild_run.child_runs
    assert grandchild_run.metadata.get("some_foo") == "some_bar"
    assert "afoo" in grandchild_run.tags  # type: ignore[operator]
    kitten_run = grandchild_run.child_runs[0]
    assert kitten_run.name == "kitten"
    assert not kitten_run.child_runs
    assert kitten_run.metadata.get("some_foo") == "some_bar"
    assert "afoo" in kitten_run.tags  # type: ignore[operator]
