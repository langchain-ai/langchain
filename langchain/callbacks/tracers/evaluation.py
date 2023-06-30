"""A tracer that runs evaluators over completed runs."""
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Optional, Sequence, Set, Union
from uuid import UUID

from langchainplus_sdk import LangChainPlusClient, RunEvaluator

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run


class EvaluatorCallbackHandler(BaseTracer):
    """A tracer that runs a run evaluator whenever a run is persisted.

    Parameters
    ----------
    evaluators : Sequence[RunEvaluator]
        The run evaluators to apply to all top level runs.
    max_workers : int, optional
        The maximum number of worker threads to use for running the evaluators.
        If not specified, it will default to the number of evaluators.
    client : LangChainPlusClient, optional
        The LangChainPlusClient instance to use for evaluating the runs.
        If not specified, a new instance will be created.
    example_id : Union[UUID, str], optional
        The example ID to be associated with the runs.

    Attributes
    ----------
    example_id : Union[UUID, None]
        The example ID associated with the runs.
    client : LangChainPlusClient
        The LangChainPlusClient instance used for evaluating the runs.
    evaluators : Sequence[RunEvaluator]
        The sequence of run evaluators to be executed.
    executor : ThreadPoolExecutor
        The thread pool executor used for running the evaluators.
    futures : Set[Future]
        The set of futures representing the running evaluators.
    """

    name = "evaluator_callback_handler"

    def __init__(
        self,
        evaluators: Sequence[RunEvaluator],
        max_workers: Optional[int] = None,
        client: Optional[LangChainPlusClient] = None,
        example_id: Optional[Union[UUID, str]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.client = client or LangChainPlusClient()
        self.evaluators = evaluators
        self.executor = ThreadPoolExecutor(
            max_workers=max(max_workers or len(evaluators), 1)
        )
        self.futures: Set[Future] = set()

    def _persist_run(self, run: Run) -> None:
        """Run the evaluator on the run.

        Parameters
        ----------
        run : Run
            The run to be evaluated.

        """
        run_ = run.copy()
        run_.reference_example_id = self.example_id
        for evaluator in self.evaluators:
            self.futures.add(
                self.executor.submit(self.client.evaluate_run, run_, evaluator)
            )

    def wait_for_futures(self) -> None:
        """Wait for all futures to complete."""
        futures = list(self.futures)
        wait(futures)
        for future in futures:
            self.futures.remove(future)
