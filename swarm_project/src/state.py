from typing import Annotated, List, TypedDict
from operator import add

class SwarmState(TypedDict):
    # Shared memory of the swarm
    messages: Annotated[List, add]
    # The current code/artifact being worked on
    artifact: str
    # Counter to prevent infinite loops
    iterations: int
