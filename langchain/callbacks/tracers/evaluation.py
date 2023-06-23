"""A tracer that runs evaluators over completed runs."""
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, List, Optional, Sequence, Set, Union
from uuid import UUID

from langchainplus_sdk import LangChainPlusClient, RunEvaluator

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run


class EvaluatorCallbackHandler(BaseTracer):
    """A tracer that runs a run evaluator whenever a

    run is persisted."""

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
        self.executor = ThreadPoolExecutor(max_workers=max_workers or len(evaluators))
        self.futures: Set[Future] = set()

    def _persist_run(self, run: Run) -> None:
        """Run the evaluator on the run."""
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
