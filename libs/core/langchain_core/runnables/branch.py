from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import (
    Runnable,
    RunnableLike,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)


class RunnableBranch(RunnableSerializable[Input, Output]):
    """Runnable that selects which branch to run based on a condition.

    The Runnable is initialized with a list of (condition, Runnable) pairs and
    a default branch.

    When operating on an input, the first condition that evaluates to True is
    selected, and the corresponding Runnable is run on the input.

    If no condition evaluates to True, the default branch is run on the input.

    Parameters:
        branches: A list of (condition, Runnable) pairs.
        default: A Runnable to run if no condition is met.

    Examples:

        .. code-block:: python

            from langchain_core.runnables import RunnableBranch

            branch = RunnableBranch(
                (lambda x: isinstance(x, str), lambda x: x.upper()),
                (lambda x: isinstance(x, int), lambda x: x + 1),
                (lambda x: isinstance(x, float), lambda x: x * 2),
                lambda x: "goodbye",
            )

            branch.invoke("hello") # "HELLO"
            branch.invoke(None) # "goodbye"
    """

    branches: Sequence[Tuple[Runnable[Input, bool], Runnable[Input, Output]]]
    default: Runnable[Input, Output]

    def __init__(
        self,
        *branches: Union[
            Tuple[
                Union[
                    Runnable[Input, bool],
                    Callable[[Input], bool],
                    Callable[[Input], Awaitable[bool]],
                ],
                RunnableLike,
            ],
            RunnableLike,  # To accommodate the default branch
        ],
    ) -> None:
        """A Runnable that runs one of two branches based on a condition.

        Args:
            *branches: A list of (condition, Runnable) pairs.
                Defaults a Runnable to run if no condition is met.

        Raises:
            ValueError: If the number of branches is less than 2.
            TypeError: If the default branch is not Runnable, Callable or Mapping.
            TypeError: If a branch is not a tuple or list.
            ValueError: If a branch is not of length 2.
        """
        if len(branches) < 2:
            raise ValueError("RunnableBranch requires at least two branches")

        default = branches[-1]

        if not isinstance(
            default,
            (Runnable, Callable, Mapping),  # type: ignore[arg-type]
        ):
            raise TypeError(
                "RunnableBranch default must be Runnable, callable or mapping."
            )

        default_ = cast(
            Runnable[Input, Output], coerce_to_runnable(cast(RunnableLike, default))
        )

        _branches = []

        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):  # type: ignore[arg-type]
                raise TypeError(
                    f"RunnableBranch branches must be "
                    f"tuples or lists, not {type(branch)}"
                )

            if not len(branch) == 2:
                raise ValueError(
                    f"RunnableBranch branches must be "
                    f"tuples or lists of length 2, not {len(branch)}"
                )
            condition, runnable = branch
            condition = cast(Runnable[Input, bool], coerce_to_runnable(condition))
            runnable = coerce_to_runnable(runnable)
            _branches.append((condition, runnable))

        super().__init__(branches=_branches, default=default_)  # type: ignore[call-arg]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """RunnableBranch is serializable if all its branches are serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        runnables = (
            [self.default]
            + [r for _, r in self.branches]
            + [r for r, _ in self.branches]
        )

        for runnable in runnables:
            if runnable.get_input_schema(config).schema().get("type") is not None:
                return runnable.get_input_schema(config)

        return super().get_input_schema(config)

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        from langchain_core.beta.runnables.context import (
            CONTEXT_CONFIG_PREFIX,
            CONTEXT_CONFIG_SUFFIX_SET,
        )

        specs = get_unique_config_specs(
            spec
            for step in (
                [self.default]
                + [r for _, r in self.branches]
                + [r for r, _ in self.branches]
            )
            for spec in step.config_specs
        )
        if any(
            s.id.startswith(CONTEXT_CONFIG_PREFIX)
            and s.id.endswith(CONTEXT_CONFIG_SUFFIX_SET)
            for s in specs
        ):
            raise ValueError("RunnableBranch cannot contain context setters.")
        return specs

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """First evaluates the condition, then delegate to true or false branch.

        Args:
            input: The input to the Runnable.
            config: The configuration for the Runnable. Defaults to None.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Returns:
            The output of the branch that was run.

        Raises:

        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = runnable.invoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = self.default.invoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)
        return output

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """Async version of invoke."""
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = await runnable.ainvoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = await self.default.ainvoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)
        return output

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """First evaluates the condition,
        then delegate to true or false branch.

        Args:
            input: The input to the Runnable.
            config: The configuration for the Runnable. Defaults to None.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            The output of the branch that was run.

        Raises:
            BaseException: If an error occurs during the execution of the Runnable.
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Optional[Output] = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    for chunk in runnable.stream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                for chunk in self.default.stream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(final_output)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """First evaluates the condition,
        then delegate to true or false branch.

        Args:
            input: The input to the Runnable.
            config: The configuration for the Runnable. Defaults to None.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            The output of the branch that was run.

        Raises:
            BaseException: If an error occurs during the execution of the Runnable.
        """
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Optional[Output] = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    async for chunk in runnable.astream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                async for chunk in self.default.astream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(final_output)
