"""Output verification wrapper for `Runnable` objects."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator  # noqa: TC003
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_core.exceptions import OutputVerificationError
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

_DEFAULT_AUDIT_SNIPPET_LEN = 10_000


class RunnableWithOutputVerification(RunnableBindingBase[Input, Output]):  # type: ignore[no-redef]
    """Wrap a `Runnable` to verify its output before returning it to callers.

    After the bound runnable completes, the optional output is passed to `verify`.
    If `verify` returns `False`, an `OutputVerificationError` is raised and the
    output is not returned downstream.

    You can record each decision by passing `audit_sink` and/or `on_audit`.

    !!! note

        `transform` / `atransform` forward to the bound runnable without running
        verification, because chunk boundaries are not guaranteed to represent
        complete logical outputs.

        `batch_as_completed` / `abatch_as_completed` use the bound runnable's
        implementation directly and do not run output verification.

    Example:
        ```python
        from langchain_core.runnables import RunnableLambda


        def is_safe(text: object) -> bool:
            return "error" not in str(text).lower()


        chain = RunnableLambda(lambda _: "ok").with_output_verification(
            verify=is_safe,
            step_name="demo",
        )
        chain.invoke({})
        ```

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    verify: Callable[[Any], bool] = Field(...)
    """Return `True` if the output may proceed, `False` to block."""

    step_name: str = "runnable"
    """Label stored in audit entries and on `OutputVerificationError`."""

    audit_sink: Any | None = None
    """If set, append one JSON-friendly dict per verification attempt.

    Use a `list` you own; this field is typed as `Any` so the same object is
    kept (Pydantic would otherwise copy `list[dict[...]]` inputs).
    """

    on_audit: Callable[[dict[str, Any]], None] | None = None
    """Optional callback invoked with the same dict appended to `audit_sink`."""

    audit_max_output_len: int = Field(
        default=_DEFAULT_AUDIT_SNIPPET_LEN,
        ge=1,
    )
    """Truncate `raw_output` in audit entries to this many characters."""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """This runnable holds callables and is not serializable."""
        return False

    def _snapshot_output(self, output: Any) -> str:
        text = str(output)
        if len(text) > self.audit_max_output_len:
            return f"{text[: self.audit_max_output_len]}...<truncated>"
        return text

    def _emit_audit(self, *, verified: bool, output: Any) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": self.step_name,
            "raw_output": self._snapshot_output(output),
            "status": "VERIFIED" if verified else "BLOCKED",
        }
        if self.audit_sink is not None:
            cast("list[Any]", self.audit_sink).append(entry)
        if self.on_audit is not None:
            self.on_audit(entry)

    def _verify_and_audit(self, output: Any) -> Output:
        ok = self.verify(output)
        if ok:
            self._emit_audit(verified=True, output=output)
            return cast("Output", output)
        self._emit_audit(verified=False, output=output)
        msg = f"Output verification failed for step {self.step_name!r}"
        raise OutputVerificationError(
            msg,
            step_name=self.step_name,
            output=output,
        )

    @staticmethod
    def _aggregate_stream_chunks(chunks: list[Output]) -> Output:
        if not chunks:
            msg = "Cannot aggregate an empty stream."
            raise ValueError(msg)
        aggregated: Output = chunks[0]
        for chunk in chunks[1:]:
            try:
                aggregated = aggregated + chunk  # type: ignore[operator]
            except TypeError:
                return chunks[-1]
        return aggregated

    @override
    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        out = super().invoke(input, config, **kwargs)
        return self._verify_and_audit(out)

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        out = await super().ainvoke(input, config, **kwargs)
        return self._verify_and_audit(out)

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        outs = super().batch(
            inputs,
            config,
            return_exceptions=return_exceptions,
            **kwargs,
        )
        if return_exceptions:
            verified_exc: list[Output | Exception] = []
            for o in outs:
                if isinstance(o, Exception):
                    verified_exc.append(o)
                else:
                    verified_exc.append(self._verify_and_audit(o))
            return cast("list[Output]", verified_exc)
        return [self._verify_and_audit(o) for o in outs]

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        outs = await super().abatch(
            inputs,
            config,
            return_exceptions=return_exceptions,
            **kwargs,
        )
        if return_exceptions:
            averified_exc: list[Output | Exception] = []
            for o in outs:
                if isinstance(o, Exception):
                    averified_exc.append(o)
                else:
                    averified_exc.append(self._verify_and_audit(o))
            return cast("list[Output]", averified_exc)
        return [self._verify_and_audit(o) for o in outs]

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        merged_config = self._merge_configs(config)
        merged_kw = {**self.kwargs, **kwargs}
        chunks: list[Output] = list(
            self.bound.stream(input, merged_config, **merged_kw),
        )
        if not chunks:
            return
        aggregated = self._aggregate_stream_chunks(chunks)
        self._verify_and_audit(aggregated)
        yield from chunks

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        merged_config = self._merge_configs(config)
        merged_kw = {**self.kwargs, **kwargs}
        chunks: list[Output] = [
            chunk
            async for chunk in self.bound.astream(input, merged_config, **merged_kw)
        ]
        if not chunks:
            return
        aggregated = self._aggregate_stream_chunks(chunks)
        self._verify_and_audit(aggregated)
        for chunk in chunks:
            yield chunk

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        return super().transform(input, config, **kwargs)

    @override
    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async for item in super().atransform(input, config, **kwargs):
            yield item


__all__ = ["RunnableWithOutputVerification"]
