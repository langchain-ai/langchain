from __future__ import annotations

from uuid import UUID

from langchain_core.pydantic_v1 import BaseModel


class RunInfo(BaseModel):
    """Class that contains metadata for a single execution of a Chain or model.

    Here for backwards compatibility with older versions of langchain_core.

    This model will likely be deprecated in the future.

    Users can acquire the run_id information from callbacks or via run_id
    information present in the astream_event API (depending on the use case).
    """

    run_id: UUID
    """A unique identifier for the model or chain run."""
