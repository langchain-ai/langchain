from collections import deque
from typing import TYPE_CHECKING, Dict, List, Union

if TYPE_CHECKING:
    import pandas as pd


class MetricsTrackerAverage:
    """Metrics Tracker Average."""

    def __init__(self, step: int):
        self.history: List[Dict[str, Union[int, float]]] = [{"step": 0, "score": 0}]
        self.step: int = step
        self.i: int = 0
        self.num: float = 0
        self.denom: float = 0

    @property
    def score(self) -> float:
        return self.num / self.denom if self.denom > 0 else 0

    def on_decision(self) -> None:
        self.denom += 1

    def on_feedback(self, score: float) -> None:
        self.num += score or 0
        self.i += 1
        if self.step > 0 and self.i % self.step == 0:
            self.history.append({"step": self.i, "score": self.score})

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(self.history)


class MetricsTrackerRollingWindow:
    """Metrics Tracker Rolling Window."""

    def __init__(self, window_size: int, step: int):
        self.history: List[Dict[str, Union[int, float]]] = [{"step": 0, "score": 0}]
        self.step: int = step
        self.i: int = 0
        self.window_size: int = window_size
        self.queue: deque = deque()
        self.sum: float = 0.0

    @property
    def score(self) -> float:
        return self.sum / len(self.queue) if len(self.queue) > 0 else 0

    def on_decision(self) -> None:
        pass

    def on_feedback(self, value: float) -> None:
        self.sum += value
        self.queue.append(value)
        self.i += 1

        if len(self.queue) > self.window_size:
            old_val = self.queue.popleft()
            self.sum -= old_val

        if self.step > 0 and self.i % self.step == 0:
            self.history.append({"step": self.i, "score": self.sum / len(self.queue)})

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(self.history)
