"""
Configuration settings for the Ticket Broker Optimization System
"""
import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///ticket_broker.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # API Keys
    spotify_client_id: Optional[str] = Field(default=None, env="SPOTIFY_CLIENT_ID")
    spotify_client_secret: Optional[str] = Field(default=None, env="SPOTIFY_CLIENT_SECRET")
    twitter_bearer_token: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    twitter_api_key: Optional[str] = Field(default=None, env="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(default=None, env="TWITTER_API_SECRET")
    twitter_access_token: Optional[str] = Field(default=None, env="TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret: Optional[str] = Field(default=None, env="TWITTER_ACCESS_TOKEN_SECRET")
    ticketmaster_api_key: Optional[str] = Field(default=None, env="TICKETMASTER_API_KEY")
    google_trends_api_key: Optional[str] = Field(default=None, env="GOOGLE_TRENDS_API_KEY")
    
    # Economic Data APIs
    fred_api_key: Optional[str] = Field(default=None, env="FRED_API_KEY")
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    
    # Scraping Configuration
    selenium_webdriver_path: str = Field(default="/usr/local/bin/chromedriver", env="SELENIUM_WEBDRIVER_PATH")
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        env="USER_AGENT"
    )
    
    # Business Rules
    minimum_profit_margin: float = Field(default=0.40, env="MINIMUM_PROFIT_MARGIN")  # 40%
    max_investment_per_event: float = Field(default=5000.0, env="MAX_INVESTMENT_PER_EVENT")
    max_total_inventory_value: float = Field(default=50000.0, env="MAX_TOTAL_INVENTORY_VALUE")
    
    # Scoring Thresholds
    high_profit_score_threshold: int = Field(default=15, env="HIGH_PROFIT_SCORE_THRESHOLD")
    moderate_profit_score_threshold: int = Field(default=10, env="MODERATE_PROFIT_SCORE_THRESHOLD")
    low_profit_score_threshold: int = Field(default=5, env="LOW_PROFIT_SCORE_THRESHOLD")
    
    # Data Refresh Intervals (in minutes)
    social_media_refresh_interval: int = Field(default=60, env="SOCIAL_MEDIA_REFRESH_INTERVAL")
    billboard_refresh_interval: int = Field(default=1440, env="BILLBOARD_REFRESH_INTERVAL")  # Daily
    economic_data_refresh_interval: int = Field(default=720, env="ECONOMIC_DATA_REFRESH_INTERVAL")  # 12 hours
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="ticket_broker.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Scoring Configuration
SCORING_WEIGHTS = {
    "high_profit_indicators": 3,
    "medium_profit_indicators": 2,
    "low_profit_indicators": 1,
    "risk_factors": -2
}

HIGH_PROFIT_INDICATORS = [
    "first_tour_in_2_years",
    "venue_under_5000_capacity",
    "sold_out_under_1_hour",
    "major_award_winner",
    "farewell_retirement_tour"
]

MEDIUM_PROFIT_INDICATORS = [
    "new_album_within_6_months",
    "premium_venue",
    "strong_local_fanbase",
    "weekend_performance",
    "holiday_special_date"
]

LOW_PROFIT_INDICATORS = [
    "established_touring_act",
    "mid_tier_venue",
    "moderate_social_buzz",
    "standard_ticket_prices",
    "regular_tour_stop"
]

RISK_FACTORS = [
    "frequent_touring",
    "large_venue_20k_plus",
    "competing_major_events",
    "declining_popularity",
    "economic_downturn"
]

# Venue Classifications
VENUE_CATEGORIES = {
    "small": {"max_capacity": 5000, "multiplier": 1.5},
    "medium": {"max_capacity": 15000, "multiplier": 1.2},
    "large": {"max_capacity": 50000, "multiplier": 1.0},
    "stadium": {"max_capacity": float('inf'), "multiplier": 0.8}
}

# Market Tier Classifications
MARKET_TIERS = {
    "tier_1": ["New York", "Los Angeles", "Chicago", "San Francisco", "Boston", "Washington DC"],
    "tier_2": ["Atlanta", "Dallas", "Houston", "Miami", "Philadelphia", "Phoenix", "Seattle"],
    "tier_3": ["Denver", "Las Vegas", "Minneapolis", "Portland", "San Diego", "Tampa"]
}