"""Pydantic models for input validation."""
from __future__ import annotations

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional


class VideoInput(BaseModel):
    url: HttpUrl = Field(..., description="YouTube or Instagram URL to process")
    language: Optional[str] = Field("en", description="Language code for processing")
    include_competitor_analysis: bool = Field(False)


class ProcessRequest(BaseModel):
    video: VideoInput
    dry_run: bool = Field(False, description="If true, avoid calling external APIs")
