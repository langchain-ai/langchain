"""A tracer that runs evaluators over completed runs."""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Set, Union
from uuid import UUID

import langsmith
from langsmith import schemas as langsmith_schemas

from langchain.callbacks import manager
from langchain.callbacks.tracers import langchain as langchain_tracer
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run

logger = logging.getLogger(__name__)


class EvaluatorCallbackHandler(BaseTracer):
    """A tracer that runs a run evaluator whenever a run is persisted.

    Parameters
    ----------
    evaluators : Sequence[RunEvaluator]
        The run evaluators to apply to all top level runs.
    max_workers : int, optional
        The maximum number of worker threads to use for running the evaluators.
        If not specified, it will default to the number of evaluators.
    client : LangSmith Client, optional
        The LangSmith client instance to use for evaluating the runs.
        If not specified, a new instance will be created.
    example_id : Union[UUID, str], optional
        The example ID to be associated with the runs.
    project_name : str, optional
        The LangSmith project name to be organize eval chain runs under.

    Attributes
    ----------
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

    name = "evaluator_callback_handler"

    def __init__(
        self,
        evaluators: Sequence[langsmith.RunEvaluator],
        max_workers: Optional[int] = None,
        client: Optional[langsmith.Client] = None,
        example_id: Optional[Union[UUID, str]] = None,
        skip_unfinished: bool = True,
        project_name: Optional[str] = "evaluators",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.client = client or langchain_tracer.get_client()
        self.evaluators = evaluators
        self.max_workers = max_workers or len(evaluators)
        self.futures: Set[Future] = set()
        self.skip_unfinished = skip_unfinished
        self.project_name = project_name
        self.logged_feedback: Dict[str, List[langsmith_schemas.Feedback]] = {}

    def _evaluate_in_project(self, run: Run, evaluator: langsmith.RunEvaluator) -> None:
        """Evaluate the run in the project.

        Parameters
        ----------
        run : Run
            The run to be evaluated.
        evaluator : RunEvaluator
            The evaluator to use for evaluating the run.

        """
        try:
            if self.project_name is None:
                feedback = self.client.evaluate_run(run, evaluator)
            with manager.tracing_v2_enabled(
                project_name=self.project_name, tags=["eval"], client=self.client
            ):
                feedback = self.client.evaluate_run(run, evaluator)
        except Exception as e:
            logger.error(
                f"Error evaluating run {run.id} with "
                f"{evaluator.__class__.__name__}: {e}",
                exc_info=True,
            )
            raise e
        example_id = str(run.reference_example_id)
        self.logged_feedback.setdefault(example_id, []).append(feedback)

    def _persist_run(self, run: Run) -> None:
        """Run the evaluator on the run.

        Parameters
        ----------
        run : Run
            The run to be evaluated.

        """
        if self.skip_unfinished and not run.outputs:
            logger.debug(f"Skipping unfinished run {run.id}")
            return
        run_ = run.copy()
        run_.reference_example_id = self.example_id
        if self.max_workers > 0:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                list(
                    executor.map(
                        self._evaluate_in_project,
                        [run_ for _ in range(len(self.evaluators))],
                        self.evaluators,
                    )
                )
        else:
            for evaluator in self.evaluators:
                self._evaluate_in_project(run_, evaluator)
