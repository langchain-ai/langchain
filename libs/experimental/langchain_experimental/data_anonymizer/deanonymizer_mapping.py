from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

MappingDataType = Dict[str, Dict[str, str]]


@dataclass
class DeanonymizerMapping:
    mapping: MappingDataType = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(str))
    )

    @property
    def data(self) -> MappingDataType:
        """Return the deanonymizer mapping"""
        return {k: dict(v) for k, v in self.mapping.items()}

    def update(self, new_mapping: MappingDataType) -> None:
        for entity_type, values in new_mapping.items():
            self.mapping[entity_type].update(values)
