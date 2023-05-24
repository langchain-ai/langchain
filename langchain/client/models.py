from datetime import datetime
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, root_validator

from langchain.callbacks.tracers.schemas import Run, RunTypeEnum


class ExampleBase(BaseModel):
    """Example base model."""

    dataset_id: UUID
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        frozen = True


class ExampleCreate(ExampleBase):
    """Example create model."""

    id: Optional[UUID]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Example(ExampleBase):
    """Example model."""

    id: UUID
    created_at: datetime
    modified_at: Optional[datetime] = Field(default=None)
    runs: List[Run] = Field(default_factory=list)


class ExampleUpdate(BaseModel):
    """Update class for Example."""

    dataset_id: Optional[UUID] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None

    class Config:
        frozen = True


class DatasetBase(BaseModel):
    """Dataset base model."""

    tenant_id: UUID
    name: str
    description: Optional[str] = None

    class Config:
        frozen = True


class DatasetCreate(DatasetBase):
    """Dataset create model."""

    id: Optional[UUID]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Dataset(DatasetBase):
    """Dataset ORM model."""

    id: UUID
    created_at: datetime
    modified_at: Optional[datetime] = Field(default=None)


class ListRunsQueryParams(BaseModel):
    """Query params for GET /runs endpoint."""

    id: Optional[List[UUID]]
    """Filter runs by id."""
    parent_run: Optional[UUID]
    """Filter runs by parent run."""
    run_type: Optional[RunTypeEnum]
    """Filter runs by type."""
    session: Optional[UUID] = Field(default=None, alias="session_id")
    """Only return runs within a session."""
    reference_example: Optional[UUID]
    """Only return runs that reference the specified dataset example."""
    execution_order: Optional[int]
    """Filter runs by execution order."""
    error: Optional[bool]
    """Whether to return only runs that errored."""
    offset: Optional[int]
    """The offset of the first run to return."""
    limit: Optional[int]
    """The maximum number of runs to return."""
    start_time: Optional[datetime] = Field(
        default=None,
        alias="start_before",
        description="Query Runs that started <= this time",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        alias="end_after",
        description="Query Runs that ended >= this time",
    )

    class Config:
        extra = "forbid"
        frozen = True

    @root_validator
    def validate_time_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that start_time <= end_time."""
        start_time = values.get("start_time")
        end_time = values.get("end_time")
        if start_time and end_time and start_time > end_time:
            raise ValueError("start_time must be <= end_time")
        return values


class APIFeedbackSource(BaseModel):
    """API feedback source."""

    type: ClassVar[str] = "api"
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        frozen = True


class FeedbackBase(BaseModel):
    """Feedback schema."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    """The time the feedback was created."""
    modified_at: datetime = Field(default_factory=datetime.utcnow)
    """The time the feedback was last modified."""
    run_id: UUID
    """The associated run ID this feedback is logged for."""
    metric_name: str
    """The feedback metric name or type."""
    metric_value: Union[float, bool, int, str]
    """Score to assign the run."""
    feedback_source: Optional[Union[APIFeedbackSource, Mapping[str, Any]]] = None
    """The source of the feedback."""

    class Config:
        frozen = True


class FeedbackCreate(FeedbackBase):
    """Schema used for creating feedback."""

    id: UUID = Field(default_factory=uuid4)

    feedback_source: APIFeedbackSource
    """The source of the feedback."""


class Feedback(FeedbackBase):
    """Schema for getting feedback."""

    id: UUID
    feedback_source: Optional[Dict] = None
    """The source of the feedback. In this case"""


class ListFeedbackQueryParams(BaseModel):
    """Query Params for listing feedbacks."""

    run: Optional[Sequence[UUID]] = None
    limit: int = 100
    offset: int = 0

    class Config:
        """Config for query params."""

        extra = "forbid"
        frozen = True
