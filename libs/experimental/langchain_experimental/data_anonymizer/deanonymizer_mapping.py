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
        """Update the deanonymizer mapping with new values
        Duplicate values will not be added
        """
        new_values_seen = set()

        for entity_type, values in new_mapping.items():
            for k, v in values.items():
                # Make sure it is not a duplicate value
                if (
                    v not in self.mapping[entity_type].values()
                    and v not in new_values_seen
                ):
                    self.mapping[entity_type][k] = v
                    new_values_seen.update({v})
