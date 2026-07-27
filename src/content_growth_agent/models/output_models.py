"""Pydantic models for agent outputs."""
from __future__ import annotations

from pydantic import BaseModel
from typing import List, Optional


class SEOOutput(BaseModel):
    titles: List[str]
    descriptions: List[str]
    keywords: List[str]


class ThumbnailOutput(BaseModel):
    texts: List[str]
    color_schemes: Optional[List[str]] = None


class CaptionOutput(BaseModel):
    instagram: str
    linkedin: str
    x_thread: List[str]


class ContentAnalysis(BaseModel):
    summary: str
    seo: SEOOutput
    thumbnail: ThumbnailOutput
    captions: CaptionOutput
    hashtags: List[str]
    calendar: Optional[List[str]] = None
    competitor_analysis: Optional[str] = None
