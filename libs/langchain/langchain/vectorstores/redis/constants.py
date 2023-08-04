from typing import Dict, List, Union, Any

# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20600},
    {"name": "searchlight", "ver": 20600},
]

# distance metrics
REDIS_DISTANCE_METRICS: List[str] = ["COSINE", "IP", "L2"]
