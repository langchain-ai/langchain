from __future__ import annotations

import hashlib
import json
import uuid


def hash_string_to_uuid(input_string: str, namespace: uuid.UUID) -> uuid.UUID:
    """Hashes a string and returns the corresponding UUID."""
    hash_value = hashlib.sha1(input_string.encode("utf-8")).hexdigest()
    return uuid.uuid5(namespace, hash_value)


def hash_nested_dict_to_uuid(data: dict, namespace: uuid.UUID) -> uuid.UUID:
    """Hashes a nested dictionary and returns the corresponding UUID."""
    serialized_data = json.dumps(data, sort_keys=True)
    hash_value = hashlib.sha1(serialized_data.encode("utf-8")).hexdigest()
    return uuid.uuid5(namespace, hash_value)
