from typing import List, TypedDict


class AddResponse(TypedDict):
    succeeded: List[str]
    failed: List[str]


class DeleteResponse(TypedDict):
    succeeded: List[str]
    failed: List[str]
