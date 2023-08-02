
from typing import Literal

# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20600},
    {"name": "searchlight", "ver": 20600},
]

# distance metrics
REDIS_DISTANCE_METRICS = Literal["COSINE", "IP", "L2"]
