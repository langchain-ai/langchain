"""A tracer that runs evaluators over completed runs."""

from __future__ import annotations

import logging
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, Optional, Union, cast
from uuid import UUID

import langsmith
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults

from langchain_core.tracers import langchain as langchain_tracer
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import _get_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)

_TRACERS: weakref.WeakSet[EvaluatorCallbackHandler] = weakref.WeakSet()


def wait_for_all_evaluators() -> None:
    """Wait for all tracers to finish."""
    global _TRACERS
    for tracer in list(_TRACERS):
        if tracer is not None:
            tracer.wait_for_futures()


class EvaluatorCallbackHandler(BaseTracer):
    """Tracer that runs a run evaluator whenever a run is persisted.

    Args:
        evaluators : Sequence[RunEvaluator]
            The run evaluators to apply to all top level runs.
        client : LangSmith Client, optional
            The LangSmith client instance to use for evaluating the runs.
            If not specified, a new instance will be created.
        example_id : Union[UUID, str], optional
            The example ID to be associated with the runs.
        project_name : str, optional
            The LangSmith project name to be organize eval chain runs under.

    Attributes:
        example_id : Union[UUID, None]
            The example ID associated with the runs.
        client : Client
            The LangSmith client instance used for evaluating the runs.
        evaluators : Sequence[RunEvaluator]
            The sequence of run evaluators to be executed.
        executor : ThreadPoolExecutor
            The thread pool executor used for running the evaluators.
        futures : Set[Future]
            The set of futures representing the running evaluators.
        skip_unfinished : bool
            Whether to skip runs that are not finished or raised
            an error.
        project_name : Optional[str]
            The LangSmith project name to be organize eval chain runs under.
    """

    name: str = "evaluator_callback_handler"

    def __init__(
        self,
        evaluators: Sequence[langsmith.RunEvaluator],
        client: Optional[langsmith.Client] = None,
        example_id: Optional[Union[UUID, str]] = None,
        skip_unfinished: bool = True,
        project_name: Optional[str] = "evaluators",
        max_concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.client = client or langchain_tracer.get_client()
        self.evaluators = evaluators
        if max_concurrency is None:
            self.executor: Optional[ThreadPoolExecutor] = _get_executor()
        elif max_concurrency > 0:
            self.executor = ThreadPoolExecutor(max_workers=max_concurrency)
            weakref.finalize(
                self,
                lambda: cast(ThreadPoolExecutor, self.executor).shutdown(wait=True),
            )
        else:
            self.executor = None
        self.futures: weakref.WeakSet[Future] = weakref.WeakSet()
        self.skip_unfinished = skip_unfinished
        self.project_name = project_name
        self.logged_eval_results: dict[tuple[str, str], list[EvaluationResult]] = {}
        self.lock = threading.Lock()
        global _TRACERS
        _TRACERS.add(self)

    def _evaluate_in_project(self, run: Run, evaluator: langsmith.RunEvaluator) -> None:
        """Evaluate the run in the project.

        Args:
        ----------
        run : Run
            The run to be evaluated.
        evaluator : RunEvaluator
            The evaluator to use for evaluating the run.

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
                    run,  # type: ignore
                    example=reference_example,
                )
                eval_results = self._log_evaluation_feedback(
                    evaluation_result,
                    run,
                    source_run_id=cb.latest_run.id if cb.latest_run else None,
                )
        except Exception as e:
            logger.error(
                f"Error evaluating run {run.id} with "
                f"{evaluator.__class__.__name__}: {repr(e)}",
                exc_info=True,
            )
            raise
        example_id = str(run.reference_example_id)
        with self.lock:
            for res in eval_results:
                run_id = str(getattr(res, "target_run_id", run.id))
                self.logged_eval_results.setdefault((run_id, example_id), []).append(
                    res
                )

    def _select_eval_results(
        self,
        results: Union[EvaluationResult, EvaluationResults],
    ) -> list[EvaluationResult]:
        if isinstance(results, EvaluationResult):
            results_ = [results]
        elif isinstance(results, dict) and "results" in results:
            results_ = cast(list[EvaluationResult], results["results"])
        else:
            msg = (
                f"Invalid evaluation result type {type(results)}."
                " Expected EvaluationResult or EvaluationResults."
            )
            raise TypeError(msg)
        return results_

    def _log_evaluation_feedback(
        self,
        evaluator_response: Union[EvaluationResult, EvaluationResults],
        run: Run,
        source_run_id: Optional[UUID] = None,
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
        ----------
        run : Run
            The run to be evaluated.

        """
        if self.skip_unfinished and not run.outputs:
            logger.debug(f"Skipping unfinished run {run.id}")
            return
        run_ = run.copy()
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
