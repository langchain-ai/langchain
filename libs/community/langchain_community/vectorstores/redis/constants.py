from typing import Any, Dict, List

import numpy as np

# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20600},
    {"name": "searchlight", "ver": 20600},
]

# distance metrics
REDIS_DISTANCE_METRICS: List[str] = ["COSINE", "IP", "L2"]

# supported vector datatypes
REDIS_VECTOR_DTYPE_MAP: Dict[str, Any] = {
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}

REDIS_TAG_SEPARATOR = ","
