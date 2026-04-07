from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import threading
import uuid
from inspect import isasyncgenfunction
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import MagicMock, patch

import pytest
from langsmith import Client, RunTree, get_current_run_tree, traceable
from langsmith.run_helpers import tracing_context
from langsmith.utils import get_env_var

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.runnables.base import RunnableLambda, RunnableParallel
from langchain_core.tracers.langchain import LangChainTracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine, Generator, Mapping


def _get_posts(client: Client) -> list[dict[str, Any]]:
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
    metadata: Mapping[str, str] | None = None,
) -> LangChainTracer:
    mock_session = MagicMock()
    mock_client_ = Client(
        session=mock_session, api_key="test", auto_batch_tracing=False
    )
    return LangChainTracer(
        client=mock_client_, project_name=project_name, tags=tags, metadata=metadata
    )


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


def test_inheritable_metadata_respects_explicit_metadata_with_tracing_context() -> None:
    """Tracer defaults fill missing keys while run metadata keeps precedence."""
    tracer = _create_tracer_with_mocked_client()

    @RunnableLambda
    def my_func(x: int) -> int:
        return x

    callbacks = CallbackManager.configure(
        inheritable_callbacks=[tracer],
        langsmith_inheritable_metadata={
            "tenant": "from_tracer",
            "shared": "from_tracer",
        },
    )
    with tracing_context(enabled=True, client=tracer.client):
        my_func.invoke(
            1,
            {
                "callbacks": callbacks,
                "metadata": {"shared": "from_run", "explicit": "from_run"},
            },
        )

    posts = _get_posts(tracer.client)
    assert len(posts) == 1
    metadata = posts[0].get("extra", {}).get("metadata", {})
    assert metadata["tenant"] == "from_tracer"
    assert metadata["shared"] == "from_run"
    assert metadata["explicit"] == "from_run"


def test_config_traceable_handoff() -> None:
    if hasattr(get_env_var, "cache_clear"):
        get_env_var.cache_clear()  # type: ignore[attr-defined]
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

    @RunnableLambda
    async def my_grandchild_function(a: int) -> int:
        return my_great_grandchild_function.invoke(a)

    @traceable
    async def my_child_function(a: int) -> int:
        return await my_grandchild_function.ainvoke(a) * 3

    @traceable()
    async def my_function(a: int) -> int:
        return await my_child_function(a)

    async def my_parent_function(a: int) -> int:
        return await my_function(a)

    my_parent_runnable = RunnableLambda(my_parent_function)
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

    if hasattr(get_env_var, "cache_clear"):
        get_env_var.cache_clear()  # type: ignore[attr-defined]
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

        def after(x: dict[str, Any]) -> int:
            return int(x["chain_result"])

        sequence = before | parallel | after
        if isasyncgenfunction(other_thing):

            @RunnableLambda
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
    async def ainvoke(
        parent: RunnableLambda[int, int], cb: list[BaseCallbackHandler]
    ) -> int:
        return await parent.ainvoke(1, {"callbacks": cb})

    @staticmethod
    async def astream(
        parent: RunnableLambda[int, int], cb: list[BaseCallbackHandler]
    ) -> int:
        return [res async for res in parent.astream(1, {"callbacks": cb})][-1]

    @staticmethod
    async def abatch(
        parent: RunnableLambda[int, int], cb: list[BaseCallbackHandler]
    ) -> int:
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
    grandchild_run = None
    kitten_run = None

    @traceable
    def kitten(x: str) -> str:
        nonlocal kitten_run
        kitten_run = get_current_run_tree()
        return x

    @RunnableLambda
    def grandchild(x: str) -> str:
        nonlocal grandchild_run
        grandchild_run = get_current_run_tree()
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

        def collect_langsmith_run(run: RunTree) -> None:
            collected[str(run.id)] = run

        def collect_tracer_run(_: LangChainTracer, run: RunTree) -> None:
            collected[str(run.id)] = run

        if parent_type == "ls":

            @traceable
            def parent() -> str:
                return child.invoke("foo")

            assert (
                parent(langsmith_extra={"on_end": collect_langsmith_run, "run_id": rid}) == "foo"
            )
            assert collected

        else:

            @RunnableLambda
            def parent(_: Any) -> str:
                return child.invoke("foo")

            tracer = LangChainTracer()
            with patch.object(LangChainTracer, "_persist_run", new=collect_tracer_run):
                assert parent.invoke(..., {"run_id": rid, "callbacks": [tracer]}) == "foo"  # type: ignore[attr-defined]
    run = collected.get(str(rid))

    assert run is not None
    assert run.name == "parent"
    assert run.child_runs
    child_run = run.child_runs[0]
    assert child_run.name == "child"
    assert isinstance(grandchild_run, RunTree)
    assert grandchild_run.name == "grandchild"
    assert grandchild_run.metadata.get("some_foo") == "some_bar"
    assert "afoo" in grandchild_run.tags  # type: ignore[operator]
    assert isinstance(kitten_run, RunTree)
    assert kitten_run.name == "kitten"
    assert not kitten_run.child_runs
    assert kitten_run.metadata.get("some_foo") == "some_bar"
    assert "afoo" in kitten_run.tags  # type: ignore[operator]
    assert grandchild_run is not None
    assert kitten_run.dotted_order.startswith(grandchild_run.dotted_order)


class TestTracerMetadataThroughInvoke:
    """Tests for tracer metadata merging through invoke calls."""

    def test_tracer_metadata_applied_to_all_runs(self) -> None:
        """Tracer metadata appears on every run when no config metadata is set."""
        tracer = _create_tracer_with_mocked_client(
            metadata={"env": "prod", "service": "api"}
        )

        @RunnableLambda
        def child(x: int) -> int:
            return x + 1

        @RunnableLambda
        def parent(x: int) -> int:
            return child.invoke(x)

        parent.invoke(1, {"callbacks": [tracer]})

        posts = _get_posts(tracer.client)
        assert len(posts) == 2
        for post in posts:
            md = post.get("extra", {}).get("metadata", {})
            assert md.get("env") == "prod", f"run {post['name']} missing env"
            assert md.get("service") == "api", f"run {post['name']} missing service"

    def test_config_metadata_takes_precedence(self) -> None:
        """Config metadata wins over tracer metadata for overlapping keys."""
        tracer = _create_tracer_with_mocked_client(
            metadata={"env": "prod", "tracer_only": "yes"}
        )

        @RunnableLambda
        def my_func(x: int) -> int:
            return x

        my_func.invoke(
            1,
            {
                "callbacks": [tracer],
                "metadata": {"env": "staging", "config_only": "yes"},
            },
        )

        posts = _get_posts(tracer.client)
        assert len(posts) == 1
        md = posts[0].get("extra", {}).get("metadata", {})
        # Config wins for overlapping key
        assert md["env"] == "staging"
        # Both non-overlapping keys are present
        assert md["tracer_only"] == "yes"
        assert md["config_only"] == "yes"

    def test_nested_calls_inherit_config_metadata(self) -> None:
        """Child runs inherit config metadata; tracer metadata fills gaps."""
        tracer = _create_tracer_with_mocked_client(
            metadata={"tracer_key": "tracer_val"}
        )

        @RunnableLambda
        def child(x: int) -> int:
            return x + 1

        @RunnableLambda
        def parent(x: int) -> int:
            return child.invoke(x)

        parent.invoke(
            1,
            {
                "callbacks": [tracer],
                "metadata": {"config_key": "config_val"},
            },
        )

        posts = _get_posts(tracer.client)
        assert len(posts) == 2
        name_to_md = {
            post["name"]: post.get("extra", {}).get("metadata", {}) for post in posts
        }
        # Both parent and child should have config metadata (inherited)
        # and tracer metadata (patched in)
        for name, md in name_to_md.items():
            assert md.get("config_key") == "config_val", f"{name} missing config_key"
            assert md.get("tracer_key") == "tracer_val", f"{name} missing tracer_key"

    def test_tracer_metadata_not_applied_to_sibling_handlers(self) -> None:
        """Tracer metadata is not applied to other callback handlers.

        `_patch_missing_metadata` copies the metadata dict before patching,
        so the callback manager's shared metadata dict is not mutated.
        Other handlers should only see config metadata, not tracer metadata.
        """
        tracer = _create_tracer_with_mocked_client(
            metadata={"tracer_key": "tracer_val"}
        )

        received_metadata: list[dict[str, Any]] = []

        class MetadataCapture(BaseCallbackHandler):
            """Callback handler that records metadata from chain events."""

            def on_chain_start(self, *_args: Any, **kwargs: Any) -> None:
                received_metadata.append(dict(kwargs.get("metadata", {})))

        capture = MetadataCapture()

        @RunnableLambda
        def my_func(x: int) -> int:
            return x

        my_func.invoke(
            1,
            {
                "callbacks": [tracer, capture],
                "metadata": {"shared_key": "shared_val"},
            },
        )

        assert len(received_metadata) >= 1
        for md in received_metadata:
            assert md["shared_key"] == "shared_val"
            assert "tracer_key" not in md

        # But the posted run DOES have tracer metadata
        posts = _get_posts(tracer.client)
        assert len(posts) >= 1
        for post in posts:
            post_md = post.get("extra", {}).get("metadata", {})
            assert post_md["shared_key"] == "shared_val"
            assert post_md["tracer_key"] == "tracer_val"

    def test_tracer_metadata_with_no_config_metadata(self) -> None:
        """When no config metadata is set, tracer metadata is the sole source."""
        tracer = _create_tracer_with_mocked_client(
            metadata={"only_from_tracer": "value"}
        )

        @RunnableLambda
        def my_func(x: int) -> int:
            return x

        my_func.invoke(1, {"callbacks": [tracer]})

        posts = _get_posts(tracer.client)
        assert len(posts) == 1
        md = posts[0].get("extra", {}).get("metadata", {})
        assert md["only_from_tracer"] == "value"

    def test_empty_tracer_metadata_does_not_interfere(self) -> None:
        """Tracer with no metadata does not interfere with config metadata."""
        tracer = _create_tracer_with_mocked_client(metadata=None)

        @RunnableLambda
        def my_func(x: int) -> int:
            return x

        my_func.invoke(
            1,
            {"callbacks": [tracer], "metadata": {"config_key": "config_val"}},
        )

        posts = _get_posts(tracer.client)
        assert len(posts) == 1
        md = posts[0].get("extra", {}).get("metadata", {})
        assert md["config_key"] == "config_val"


def test_inheritable_metadata_nested_runs_preserve_parent_child_shape() -> None:
    """Concurrent nested runs keep parent-child linkage within each invocation."""
    tracer = _create_tracer_with_mocked_client()
    barrier = threading.Barrier(2)

    @RunnableLambda
    def child(x: int) -> int:
        barrier.wait()
        return x + 1

    @RunnableLambda
    def parent(x: int) -> int:
        return child.invoke(x)

    def invoke_for_tenant(tenant: str, value: int) -> int:
        callbacks = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"tenant": tenant},
        )
        return parent.invoke(value, {"callbacks": callbacks})

    threads = [
        threading.Thread(target=invoke_for_tenant, args=("alpha", 1)),
        threading.Thread(target=invoke_for_tenant, args=("beta", 2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    posts = _get_posts(tracer.client)
    assert len(posts) == 4
    parents = [post for post in posts if post["name"] == "parent"]
    children = [post for post in posts if post["name"] == "child"]
    assert len(parents) == 2
    assert len(children) == 2
    parent_ids = {parent["id"] for parent in parents}
    assert {child["parent_run_id"] for child in children} == parent_ids
    assert {
        post.get("extra", {}).get("metadata", {}).get("tenant") for post in posts
    } == {
        "alpha",
        "beta",
    }


def test_inheritable_metadata_parallel_children_keep_tenant_isolation() -> None:
    """Concurrent roots with parallel child runs keep tenant metadata isolated."""
    tracer = _create_tracer_with_mocked_client()
    barrier = threading.Barrier(4)

    @RunnableLambda
    def add_one(x: int) -> int:
        barrier.wait()
        return x + 1

    @RunnableLambda
    def add_two(x: int) -> int:
        barrier.wait()
        return x + 2

    parallel = RunnableParallel(first=add_one, second=add_two)

    def invoke_for_tenant(tenant: str, value: int) -> dict[str, int]:
        callbacks = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"tenant": tenant},
        )
        return parallel.invoke(value, {"callbacks": callbacks})

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(invoke_for_tenant, ["alpha", "beta"], [1, 2]))

    posts = _get_posts(tracer.client)
    assert len(posts) == 6
    assert {
        post.get("extra", {}).get("metadata", {}).get("tenant") for post in posts
    } == {
        "alpha",
        "beta",
    }
    posts_by_trace: dict[str, list[dict[str, Any]]] = {}
    for post in posts:
        posts_by_trace.setdefault(post["trace_id"], []).append(post)
    assert len(posts_by_trace) == 2
    assert all(len(trace_posts) == 3 for trace_posts in posts_by_trace.values())


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Asyncio context vars require Python 3.11+"
)
async def test_langsmith_inheritable_metadata_mixed_sync_async_managers_isolated() -> (
    None
):
    """Sync and async manager configure paths can overlap without metadata sharing."""
    tracer = _create_tracer_with_mocked_client()

    @RunnableLambda
    async def async_runnable(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    @RunnableLambda
    def sync_runnable(x: int) -> int:
        return x + 1

    async def run_sync() -> int:
        callbacks = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"path": "sync"},
        )
        return await asyncio.to_thread(
            sync_runnable.invoke, 1, {"callbacks": callbacks}
        )

    async def run_async() -> int:
        callbacks = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"path": "async"},
        )
        return await async_runnable.ainvoke(1, {"callbacks": callbacks})

    await asyncio.gather(run_sync(), run_async())

    posts = _get_posts(tracer.client)
    assert len(posts) == 2
    assert {
        post.get("extra", {}).get("metadata", {}).get("path") for post in posts
    } == {
        "sync",
        "async",
    }


class TestLangsmithInheritableTracingDefaultsInConfigure:
    """Tests for LangSmith inheritable tracing defaults in configure."""

    def test_langsmith_inheritable_metadata_applied_via_configure(self) -> None:
        """langsmith_inheritable_metadata flows to a copied tracer."""
        tracer = _create_tracer_with_mocked_client()
        cm = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"env": "prod", "service": "api"},
        )
        lc_tracers = [h for h in cm.handlers if isinstance(h, LangChainTracer)]
        assert len(lc_tracers) == 1
        assert lc_tracers[0] is not tracer
        assert lc_tracers[0].tracing_metadata == {"env": "prod", "service": "api"}
        assert tracer.tracing_metadata is None

    def test_langsmith_inheritable_metadata_does_not_overwrite_tracer_metadata(
        self,
    ) -> None:
        """Tracer metadata takes precedence over langsmith_inheritable_metadata."""
        tracer = _create_tracer_with_mocked_client(metadata={"env": "staging"})
        cm = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"env": "prod", "service": "api"},
        )
        lc_tracer = next(h for h in cm.handlers if isinstance(h, LangChainTracer))
        assert tracer.tracing_metadata == {"env": "staging"}
        assert lc_tracer.tracing_metadata == {"env": "staging", "service": "api"}

    def test_tracing_context_metadata_merged_into_langsmith_inheritable_metadata(
        self,
    ) -> None:
        """Tracing-context metadata merges into tracer defaults.

        LangSmith metadata keeps precedence on collisions.
        """
        tracer = _create_tracer_with_mocked_client()
        with tracing_context(
            enabled=True,
            client=tracer.client,
            metadata={"trace_only": "value", "shared": "trace"},
        ):
            cm = CallbackManager.configure(
                inheritable_callbacks=[tracer],
                langsmith_inheritable_metadata={
                    "shared": "langsmith",
                    "tenant": "alpha",
                },
            )

        lc_tracer = next(h for h in cm.handlers if isinstance(h, LangChainTracer))
        assert lc_tracer.tracing_metadata == {
            "trace_only": "value",
            "shared": "langsmith",
            "tenant": "alpha",
        }

    def test_langsmith_inheritable_metadata_end_to_end(self) -> None:
        """langsmith_inheritable_metadata in configure propagates to posted runs."""
        tracer = _create_tracer_with_mocked_client()

        @RunnableLambda
        def my_func(x: int) -> int:
            return x

        # Use langsmith_inheritable_metadata through the config callbacks path
        cm = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"env": "prod"},
        )
        my_func.invoke(1, {"callbacks": cm})

        posts = _get_posts(tracer.client)
        assert len(posts) == 1
        md = posts[0].get("extra", {}).get("metadata", {})
        assert md["env"] == "prod"

    def test_langsmith_inheritable_metadata_does_not_affect_non_tracer_handlers(
        self,
    ) -> None:
        """langsmith_inheritable_metadata only applies to tracers."""
        tracer = _create_tracer_with_mocked_client()

        received_metadata: list[dict[str, Any]] = []

        class MetadataCapture(BaseCallbackHandler):
            def on_chain_start(self, *_args: Any, **kwargs: Any) -> None:
                received_metadata.append(dict(kwargs.get("metadata", {})))

        capture = MetadataCapture()
        cm = CallbackManager.configure(
            inheritable_callbacks=[tracer, capture],
            langsmith_inheritable_metadata={"tracer_only": "yes"},
        )

        @RunnableLambda
        def my_func(x: int) -> int:
            return x

        my_func.invoke(1, {"callbacks": cm})

        # Non-tracer handler should NOT see langsmith_inheritable_metadata
        assert len(received_metadata) >= 1
        for md in received_metadata:
            assert "tracer_only" not in md

        # But the tracer's posted runs SHOULD have it
        posts = _get_posts(tracer.client)
        assert len(posts) >= 1
        for post in posts:
            post_md = post.get("extra", {}).get("metadata", {})
            assert post_md["tracer_only"] == "yes"

    def test_no_langsmith_inheritable_metadata_is_noop(self) -> None:
        """Passing langsmith_inheritable_metadata=None does not alter tracer state."""
        tracer = _create_tracer_with_mocked_client()
        cm = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata=None,
        )
        lc_tracer = next(h for h in cm.handlers if isinstance(h, LangChainTracer))
        assert lc_tracer is tracer
        assert tracer.tracing_metadata is None

    def test_langsmith_inheritable_tags_applied_via_configure(self) -> None:
        """langsmith_inheritable_tags flow to a copied tracer."""
        tracer = _create_tracer_with_mocked_client()
        tracer.tags = ["existing"]
        cm = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_tags=["tenant:alpha", "existing"],
        )
        lc_tracer = next(h for h in cm.handlers if isinstance(h, LangChainTracer))
        assert lc_tracer is not tracer
        assert lc_tracer.tags == ["existing", "tenant:alpha"]
        assert tracer.tags == ["existing"]

    def test_inheritable_tags_do_not_affect_non_tracer_handlers(self) -> None:
        """langsmith_inheritable_tags only apply to tracers."""
        tracer = _create_tracer_with_mocked_client()

        received_tags: list[list[str]] = []

        class TagCapture(BaseCallbackHandler):
            def on_chain_start(self, *_args: Any, **kwargs: Any) -> None:
                received_tags.append(list(kwargs.get("tags", [])))

        capture = TagCapture()
        cm = CallbackManager.configure(
            inheritable_callbacks=[tracer, capture],
            langsmith_inheritable_tags=["tracer-only"],
        )

        @RunnableLambda
        def my_func(x: int) -> int:
            return x

        my_func.invoke(1, {"callbacks": cm})

        assert received_tags
        assert all("tracer-only" not in tags for tags in received_tags)

        posts = _get_posts(tracer.client)
        assert posts
        assert all("tracer-only" in post.get("tags", []) for post in posts)

    def test_langsmith_inheritable_metadata_copies_handlers_without_mutating_original(
        self,
    ) -> None:
        """Configured manager copies tracers and leaves the original unchanged."""
        tracer = _create_tracer_with_mocked_client()
        cm = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"env": "prod"},
        )
        handler_tracer = next(h for h in cm.handlers if isinstance(h, LangChainTracer))
        inheritable_tracer = next(
            h for h in cm.inheritable_handlers if isinstance(h, LangChainTracer)
        )
        assert handler_tracer is not tracer
        assert inheritable_tracer is not tracer
        assert tracer.tracing_metadata is None

    def test_langsmith_inheritable_metadata_configure_isolated_per_manager(
        self,
    ) -> None:
        """Separate configure calls keep tracer-only defaults isolated."""
        tracer = _create_tracer_with_mocked_client()
        alpha_manager = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"tenant": "alpha"},
        )
        beta_manager = CallbackManager.configure(
            inheritable_callbacks=[tracer],
            langsmith_inheritable_metadata={"tenant": "beta"},
        )

        alpha_tracer = next(
            handler
            for handler in alpha_manager.handlers
            if isinstance(handler, LangChainTracer)
        )
        beta_tracer = next(
            handler
            for handler in beta_manager.handlers
            if isinstance(handler, LangChainTracer)
        )

        assert tracer.tracing_metadata is None
        assert alpha_tracer is not tracer
        assert beta_tracer is not tracer
        assert alpha_tracer is not beta_tracer
        assert alpha_tracer.tracing_metadata == {"tenant": "alpha"}
        assert beta_tracer.tracing_metadata == {"tenant": "beta"}
        assert alpha_tracer.run_map is tracer.run_map
        assert beta_tracer.run_map is tracer.run_map
        assert alpha_tracer.order_map is tracer.order_map
        assert beta_tracer.order_map is tracer.order_map

    def test_inheritable_metadata_concurrent_invocations_remain_isolated(
        self,
    ) -> None:
        """Parallel invocations through copied tracers keep metadata separated."""
        tracer = _create_tracer_with_mocked_client()
        barrier = threading.Barrier(2)

        @traceable
        def traced_leaf(x: int) -> int:
            barrier.wait()
            return x

        @RunnableLambda
        def my_func(x: int) -> int:
            return traced_leaf(x)

        def invoke_for_tenant(tenant: str, value: int) -> int:
            callbacks = CallbackManager.configure(
                inheritable_callbacks=[tracer],
                langsmith_inheritable_metadata={"tenant": tenant},
            )
            return my_func.invoke(value, {"callbacks": callbacks})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(invoke_for_tenant, ["alpha", "beta"], [1, 2]))

        posts = _get_posts(tracer.client)
        assert len(posts) == 4
        assert {post["name"] for post in posts} == {"my_func", "traced_leaf"}
        my_func_posts = [post for post in posts if post["name"] == "my_func"]
        assert len(my_func_posts) == 2
        assert {
            post.get("extra", {}).get("metadata", {}).get("tenant")
            for post in my_func_posts
        } == {"alpha", "beta"}
        assert tracer.run_map == {}
        assert len(tracer.order_map) == 2
