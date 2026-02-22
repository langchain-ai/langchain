from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class UserLocation(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    country: str | None = None
    region: str | None = None
    city: str | None = None


class WebSearchOptions(BaseModel):
    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: UserLocation | None = None
    search_type: Literal["fast", "pro", "auto"] | None = None
    image_search_relevance_enhanced: bool | None = None


class MediaResponseOverrides(BaseModel):
    return_videos: bool | None = None
    return_images: bool | None = None


class MediaResponse(BaseModel):
    overrides: MediaResponseOverrides | None = None
