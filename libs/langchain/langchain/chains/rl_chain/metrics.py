from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    import pandas as pd


class MetricsTracker:
    def __init__(self, step: int):
        self._history: List[Dict[str, Union[int, float]]] = []
        self._step: int = step
        self._i: int = 0
        self._num: float = 0
        self._denom: float = 0

    @property
    def score(self) -> float:
        return self._num / self._denom if self._denom > 0 else 0

    def on_decision(self) -> None:
        self._denom += 1

    def on_feedback(self, score: Optional[float]) -> None:
        self._num += score or 0
        self._i += 1
        if self._step > 0 and self._i % self._step == 0:
            self._history.append({"step": self._i, "score": self.score})

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(self._history)
