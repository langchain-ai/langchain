from __future__ import annotations

import dataclasses
from typing import Sequence, Union

from langchain.automaton.automaton import Transition, AbstractState


@dataclasses.dataclass(frozen=True)
class History:
    records: Sequence[Union[Transition, AbstractState]] = tuple()

    def append(self, record: Union[Transition, AbstractState]) -> History:
        """Append a record to the history."""
        return dataclasses.replace(self, records=list(self.records) + [record])

    def __repr__(self):
        num_transitions = self.get_num_transitions()
        num_states = self.get_num_states()
        return "History with {} transitions and {} states".format(
            num_transitions, num_states
        )

    def __getitem__(self, item):
        return self.records[item]

    def __len__(self):
        return len(self.records)

    def get_num_transitions(self):
        """Get the number of transitions in the history."""
        return sum(1 for record in self.records if isinstance(record, Transition))

    def get_num_states(self):
        return sum(1 for record in self.records if isinstance(record, AbstractState))

    def get_transitions(self) -> Sequence[Transition]:
        return [record for record in self.records if isinstance(record, Transition)]
