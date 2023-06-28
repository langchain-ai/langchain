"""Chain pipeline where the outputs of one step feed directly into next."""
import asyncio
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from pydantic import Extra, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
)
from langchain.chains.base import Chain


class RankingItem(NamedTuple):
    rank: int
    chain: Chain
    timeout: float


class PriorityChain(Chain):
    """Chain where multiple chains are executed in a particular order
    if the preferred chains time out."""

    rankings: List[RankingItem]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_rankings(self) -> List[RankingItem]:
        """Return the rankings passed into the chain.

        :meta private:
        """
        return self.rankings

    @root_validator(pre=True)
    def validate_priority_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the correct inputs were passed in.""" ""

        rankings = values["rankings"]

        if len(rankings) < 2:
            raise ValueError("At least 2 rankings are required.")

        prev_rank = 0
        for ranking_item in rankings:
            rank = ranking_item.rank
            chain = ranking_item.chain
            timeout = ranking_item.timeout

            if not isinstance(chain, Chain):
                raise ValueError(f"Rank {rank} does not contain a valid Chain object.")
            if not isinstance(timeout, (int, float)):
                raise ValueError(f"Rank {rank} does not contain a valid timeout.")
            if not isinstance(rank, int):
                raise ValueError(
                    f"Object {str(rank)} does not contain a valid ranking."
                )

            # ensure rankings are in ascending order and sequential
            if rank != prev_rank + 1:
                raise ValueError(
                    f"Rankings are not sequential. Expected {prev_rank + 1} but got {rank}."
                )
            prev_rank = rank

        return values

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute chains concurrently, and return the highest ranking task that completes
        before timeout. Returns a dict containing the output of the preferred chain."""

        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        all_tasks = {}

        # launch all tasks concurrently first
        for ranking_item in self.rankings:
            rank = ranking_item.rank
            chain = ranking_item.chain
            timeout = ranking_item.timeout
            callbacks = _run_manager.get_child()

            task = asyncio.create_task(
                chain.acall(inputs, return_only_outputs=True, callbacks=callbacks)
            )
            all_tasks[rank] = (task, timeout)

        # wait for highest priority task - if timeout, cancel task and move onto next
        for rank in sorted(all_tasks.keys(), key=int):
            task, timeout = all_tasks[rank]
            try:
                return await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                task.cancel()
                continue

        raise asyncio.TimeoutError("All chains timed out.")
