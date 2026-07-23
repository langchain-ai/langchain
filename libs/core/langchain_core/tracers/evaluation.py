"""A tracer that runs evaluators over completed runs."""

from __future__ import annotations

import logging
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import langsmith
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langsmith.schemas import SCORE_TYPE, VALUE_TYPE

from langchain_core.tracers import langchain as langchain_tracer
from langchain_core.tracers._compat import run_copy
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import _get_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)

_TRACERS: weakref.WeakSet[EvaluatorCallbackHandler] = weakref.WeakSet()


def make_evaluation_result(
    key: str,
    *,
    score: SCORE_TYPE = None,
    value: VALUE_TYPE = None,
    comment: str | None = None,
    correction: dict | None = None,
    evaluator_info: dict | None = None,
    feedback_config: dict | None = None,
    source_run_id: UUID | str | None = None,
    target_run_id: UUID | str | None = None,
    metadata: dict | None = None,
    extra: dict | None = None,
) -> EvaluationResult:
    """Create an `EvaluationResult`, preserving all `feedback_config` fields.

    Pydantic silently strips keys that are not declared in `FeedbackConfig` when
    `feedback_config` is passed directly to the `EvaluationResult` constructor,
    because the `Union[FeedbackConfig, dict]` annotation causes Pydantic to resolve
    the value against the `TypedDict` branch first and discard unknown keys.

    This factory constructs the object without `feedback_config`, then assigns it
    after initialisation so that the raw ``dict`` is stored intact.

    Args:
        key: The aspect, metric name, or label for this evaluation.
        score: The numeric score for this evaluation.
        value: The value for this evaluation, if not numeric.
        comment: An explanation regarding the evaluation.
        correction: What the correct value should be, if applicable.
        evaluator_info: Additional information about the evaluator.
        feedback_config: The configuration used to generate this feedback.
            All keys are preserved regardless of whether they appear in
            `FeedbackConfig`.
        source_run_id: The ID of the trace of the evaluator itself.
        target_run_id: The ID of the trace this evaluation is applied to.
        metadata: Arbitrary metadata attached to the evaluation.
        extra: Metadata for the evaluator run.

    Returns:
        An `EvaluationResult` with `feedback_config` stored without stripping.

    Example:
        .. code-block:: python

            from langchain_core.tracers.evaluation import make_evaluation_result

            result = make_evaluation_result(
                key="sentiment",
                value="positive",
                feedback_config={"threshold": 1.0},
            )
            print(result.feedback_config)  # {"threshold": 1.0}
    """
    result = EvaluationResult(
        key=key,
        score=score,
        value=value,
        comment=comment,
        correction=correction,
        evaluator_info=evaluator_info or {},
        source_run_id=source_run_id,
        target_run_id=target_run_id,
        metadata=metadata,
        extra=extra,
    )
    if feedback_config is not None:
        # Assign after __init__ to bypass Pydantic's Union[FeedbackConfig, dict]
        # resolution, which silently strips keys not declared in FeedbackConfig.
        result.feedback_config = feedback_config
    return result


def wait_for_all_evaluators() -> None:
    """Wait for all tracers to finish."""
    for tracer in list(_TRACERS):
        if tracer is not None:
            tracer.wait_for_futures()


class EvaluatorCallbackHandler(BaseTracer):
    """Tracer that runs a run evaluator whenever a run is persisted.

    Attributes:
        client: The LangSmith client instance used for evaluating the runs.
    """

    name: str = "evaluator_callback_handler"

    example_id: UUID | None = None
    """The example ID associated with the runs."""

    client: langsmith.Client
    """The LangSmith client instance used for evaluating the runs."""

    evaluators: Sequence[langsmith.RunEvaluator] = ()
    """The sequence of run evaluators to be executed."""

    executor: ThreadPoolExecutor | None = None
    """The thread pool executor used for running the evaluators."""

    futures: weakref.WeakSet[Future[None]] = weakref.WeakSet()
    """The set of futures representing the running evaluators."""

    skip_unfinished: bool = True
    """Whether to skip runs that are not finished or raised an error."""

    project_name: str | None = None
    """The LangSmith project name to be organize eval chain runs under."""

    logged_eval_results: dict[tuple[str, str], list[EvaluationResult]]

    lock: threading.Lock

    def __init__(
        self,
        evaluators: Sequence[langsmith.RunEvaluator],
        client: langsmith.Client | None = None,
        example_id: UUID | str | None = None,
        skip_unfinished: bool = True,  # noqa: FBT001,FBT002
        project_name: str | None = "evaluators",
        max_concurrency: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an EvaluatorCallbackHandler.

        Args:
            evaluators: The run evaluators to apply to all top level runs.
            client: The LangSmith client instance to use for evaluating the runs.

                If not specified, a new instance will be created.
            example_id: The example ID to be associated with the runs.
            skip_unfinished: Whether to skip unfinished runs.
            project_name: The LangSmith project name to be organize eval chain runs
                under.
            max_concurrency: The maximum number of concurrent evaluators to run.
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.client = client or langchain_tracer.get_client()
        self.evaluators = evaluators
        if max_concurrency is None:
            self.executor = _get_executor()
        elif max_concurrency > 0:
            self.executor = ThreadPoolExecutor(max_workers=max_concurrency)
            weakref.finalize(
                self,
                lambda: cast("ThreadPoolExecutor", self.executor).shutdown(wait=True),
            )
        else:
            self.executor = None
        self.futures = weakref.WeakSet[Future[None]]()
        self.skip_unfinished = skip_unfinished
        self.project_name = project_name
        self.logged_eval_results = {}
        self.lock = threading.Lock()
        _TRACERS.add(self)

    def _evaluate_in_project(self, run: Run, evaluator: langsmith.RunEvaluator) -> None:
        """Evaluate the run in the project.

        Args:
            run: The run to be evaluated.
            evaluator: The evaluator to use for evaluating the run.
        """
        try:
            if self.project_name is None:
                eval_result = self.client.evaluate_run(run, evaluator)
                eval_results = [eval_result]
            with tracing_v2_enabled(
                project_name=self.project_name, tags=["eval"], client=self.client
            ) as cb:
                reference_example = (
                    self.client.read_example(run.reference_example_id)
                    if run.reference_example_id
                    else None
                )
                evaluation_result = evaluator.evaluate_run(
                    # This is subclass, but getting errors for some reason
                    run,  # type: ignore[arg-type]
                    example=reference_example,
                )
                eval_results = self._log_evaluation_feedback(
                    evaluation_result,
                    run,
                    source_run_id=cb.latest_run.id if cb.latest_run else None,
                )
        except Exception:
            logger.exception(
                "Error evaluating run %s with %s",
                run.id,
                evaluator.__class__.__name__,
            )
            raise
        example_id = str(run.reference_example_id)
        with self.lock:
            for res in eval_results:
                run_id = str(getattr(res, "target_run_id", run.id))
                self.logged_eval_results.setdefault((run_id, example_id), []).append(
                    res
                )

    @staticmethod
    def _select_eval_results(
        results: EvaluationResult | EvaluationResults,
    ) -> list[EvaluationResult]:
        if isinstance(results, EvaluationResult):
            results_ = [results]
        elif isinstance(results, dict) and "results" in results:
            results_ = results["results"]
        else:
            msg = (
                f"Invalid evaluation result type {type(results)}."
                " Expected EvaluationResult or EvaluationResults."
            )
            raise TypeError(msg)
        return results_

    def _log_evaluation_feedback(
        self,
        evaluator_response: EvaluationResult | EvaluationResults,
        run: Run,
        source_run_id: UUID | None = None,
    ) -> list[EvaluationResult]:
        results = self._select_eval_results(evaluator_response)
        for res in results:
            source_info_: dict[str, Any] = {}
            if res.evaluator_info:
                source_info_ = {**res.evaluator_info, **source_info_}
            run_id_ = getattr(res, "target_run_id", None)
            if run_id_ is None:
                run_id_ = run.id
            self.client.create_feedback(
                run_id_,
                res.key,
                score=res.score,
                value=res.value,
                comment=res.comment,
                correction=res.correction,
                source_info=source_info_,
                source_run_id=res.source_run_id or source_run_id,
                feedback_source_type=langsmith.schemas.FeedbackSourceType.MODEL,
            )
        return results

    def _persist_run(self, run: Run) -> None:
        """Run the evaluator on the run.

        Args:
            run: The run to be evaluated.
        """
        if self.skip_unfinished and not run.outputs:
            logger.debug("Skipping unfinished run %s", run.id)
            return
        run_ = run_copy(run)
        run_.reference_example_id = self.example_id
        for evaluator in self.evaluators:
            if self.executor is None:
                self._evaluate_in_project(run_, evaluator)
            else:
                self.futures.add(
                    self.executor.submit(self._evaluate_in_project, run_, evaluator)
                )

    def wait_for_futures(self) -> None:
        """Wait for all futures to complete."""
        wait(self.futures)
