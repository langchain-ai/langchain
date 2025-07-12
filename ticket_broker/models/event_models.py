"""
Data models for the Ticket Broker Optimization System
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field


Base = declarative_base()


class EventStatus(str, Enum):
    RESEARCHING = "researching"
    APPROVED = "approved"
    REJECTED = "rejected"
    PURCHASED = "purchased"
    LISTED = "listed"
    SOLD = "sold"


class RecommendationLevel(str, Enum):
    STRONG_BUY = "strong_buy"
    SELECTIVE_BUY = "selective_buy"
    AVOID = "avoid"
    HIGH_RISK = "high_risk"


# SQLAlchemy Models
class Artist(Base):
    __tablename__ = "artists"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    spotify_id = Column(String(255), unique=True)
    genre = Column(String(100))
    monthly_listeners = Column(Integer)
    follower_count = Column(Integer)
    popularity_score = Column(Integer)
    last_album_date = Column(DateTime)
    career_stage = Column(String(50))  # rising, established, declining
    tour_frequency = Column(Float)  # tours per year
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    events = relationship("Event", back_populates="artist")
    social_metrics = relationship("SocialMetrics", back_populates="artist")


class Venue(Base):
    __tablename__ = "venues"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    city = Column(String(100), nullable=False)
    state = Column(String(50))
    country = Column(String(50), default="USA")
    capacity = Column(Integer)
    venue_type = Column(String(50))  # arena, theater, stadium, etc.
    market_tier = Column(String(20))  # tier_1, tier_2, tier_3
    prestige_level = Column(String(20))  # premium, standard, budget
    parking_availability = Column(Boolean, default=True)
    public_transport_accessible = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    events = relationship("Event", back_populates="venue")


class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    artist_id = Column(Integer, ForeignKey("artists.id"), nullable=False)
    venue_id = Column(Integer, ForeignKey("venues.id"), nullable=False)
    event_date = Column(DateTime, nullable=False)
    on_sale_date = Column(DateTime)
    face_value_min = Column(Float)
    face_value_max = Column(Float)
    estimated_resale_min = Column(Float)
    estimated_resale_max = Column(Float)
    is_ticketmaster = Column(Boolean, default=False)
    is_sold_out = Column(Boolean, default=False)
    sellout_time_minutes = Column(Integer)
    tour_name = Column(String(255))
    is_farewell_tour = Column(Boolean, default=False)
    status = Column(String(20), default=EventStatus.RESEARCHING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    artist = relationship("Artist", back_populates="events")
    venue = relationship("Venue", back_populates="events")
    scoring_results = relationship("ScoringResult", back_populates="event")
    purchases = relationship("TicketPurchase", back_populates="event")


class ScoringResult(Base):
    __tablename__ = "scoring_results"
    
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    total_score = Column(Integer, nullable=False)
    recommendation = Column(String(20), nullable=False)  # RecommendationLevel enum
    high_profit_indicators = Column(JSON)  # List of triggered indicators
    medium_profit_indicators = Column(JSON)
    low_profit_indicators = Column(JSON)
    risk_factors = Column(JSON)
    detailed_analysis = Column(JSON)  # Detailed scoring breakdown
    confidence_level = Column(Float)  # 0.0 to 1.0
    created_at = Column(DateTime, default=datetime.utcnow)
    
    event = relationship("Event", back_populates="scoring_results")


class SocialMetrics(Base):
    __tablename__ = "social_metrics"
    
    id = Column(Integer, primary_key=True)
    artist_id = Column(Integer, ForeignKey("artists.id"), nullable=False)
    platform = Column(String(50), nullable=False)  # twitter, instagram, tiktok, etc.
    follower_count = Column(Integer)
    engagement_rate = Column(Float)
    mention_count_24h = Column(Integer)
    sentiment_score = Column(Float)  # -1.0 to 1.0
    trending_keywords = Column(JSON)
    collected_at = Column(DateTime, default=datetime.utcnow)
    
    artist = relationship("Artist", back_populates="social_metrics")


class EconomicIndicator(Base):
    __tablename__ = "economic_indicators"
    
    id = Column(Integer, primary_key=True)
    city = Column(String(100), nullable=False)
    state = Column(String(50))
    unemployment_rate = Column(Float)
    median_income = Column(Float)
    consumer_confidence = Column(Float)
    entertainment_spending_index = Column(Float)
    gas_prices = Column(Float)
    date_recorded = Column(DateTime, default=datetime.utcnow)


class TicketPurchase(Base):
    __tablename__ = "ticket_purchases"
    
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    section = Column(String(50))
    row = Column(String(10))
    seat_numbers = Column(String(100))
    purchase_price = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)
    total_cost = Column(Float, nullable=False)
    listing_price = Column(Float)
    sale_price = Column(Float)
    profit_loss = Column(Float)
    platform_purchased = Column(String(50))  # stubhub, seatgeek, etc.
    platform_sold = Column(String(50))
    purchased_at = Column(DateTime, default=datetime.utcnow)
    sold_at = Column(DateTime)
    
    event = relationship("Event", back_populates="purchases")


# Pydantic Models for API
class ArtistBase(BaseModel):
    name: str
    genre: Optional[str] = None
    spotify_id: Optional[str] = None


class ArtistCreate(ArtistBase):
    pass


class ArtistResponse(ArtistBase):
    id: int
    monthly_listeners: Optional[int] = None
    follower_count: Optional[int] = None
    popularity_score: Optional[int] = None
    career_stage: Optional[str] = None
    tour_frequency: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class VenueBase(BaseModel):
    name: str
    city: str
    state: Optional[str] = None
    country: str = "USA"
    capacity: Optional[int] = None
    venue_type: Optional[str] = None


class VenueCreate(VenueBase):
    pass


class VenueResponse(VenueBase):
    id: int
    market_tier: Optional[str] = None
    prestige_level: Optional[str] = None
    parking_availability: bool = True
    public_transport_accessible: bool = True
    
    class Config:
        from_attributes = True


class EventBase(BaseModel):
    name: str
    event_date: datetime
    on_sale_date: Optional[datetime] = None
    face_value_min: Optional[float] = None
    face_value_max: Optional[float] = None
    tour_name: Optional[str] = None
    is_farewell_tour: bool = False


class EventCreate(EventBase):
    artist_id: int
    venue_id: int


class EventResponse(EventBase):
    id: int
    artist_id: int
    venue_id: int
    estimated_resale_min: Optional[float] = None
    estimated_resale_max: Optional[float] = None
    is_ticketmaster: bool = False
    is_sold_out: bool = False
    sellout_time_minutes: Optional[int] = None
    status: EventStatus
    created_at: datetime
    artist: Optional[ArtistResponse] = None
    venue: Optional[VenueResponse] = None
    
    class Config:
        from_attributes = True


class ScoringResultBase(BaseModel):
    total_score: int
    recommendation: RecommendationLevel
    confidence_level: float


class ScoringResultCreate(ScoringResultBase):
    event_id: int
    high_profit_indicators: List[str] = []
    medium_profit_indicators: List[str] = []
    low_profit_indicators: List[str] = []
    risk_factors: List[str] = []
    detailed_analysis: Dict[str, Any] = {}


class ScoringResultResponse(ScoringResultBase):
    id: int
    event_id: int
    high_profit_indicators: List[str]
    medium_profit_indicators: List[str]
    low_profit_indicators: List[str]
    risk_factors: List[str]
    detailed_analysis: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True


class MarketAnalysis(BaseModel):
    """Comprehensive market analysis for an event"""
    event_id: int
    artist_popularity_trend: str  # rising, stable, declining
    social_media_buzz: float  # 0.0 to 1.0
    local_market_strength: float  # 0.0 to 1.0
    competition_level: float  # 0.0 to 1.0
    economic_conditions: float  # 0.0 to 1.0
    venue_premium: float  # multiplier
    timing_score: float  # 0.0 to 1.0
    overall_demand_prediction: float  # 0.0 to 1.0
    suggested_purchase_limit: float  # dollar amount
    risk_assessment: str  # low, medium, high


class PurchaseRecommendation(BaseModel):
    """Final purchase recommendation with reasoning"""
    event_id: int
    recommendation: RecommendationLevel
    confidence_score: float
    total_score: int
    expected_profit_margin: float
    max_recommended_investment: float
    key_factors: List[str]
    risk_factors: List[str]
    reasoning: str
    action_required_by: Optional[datetime] = None