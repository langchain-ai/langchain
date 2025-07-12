"""
Ticket Broker Optimization System

A comprehensive system for analyzing live event tickets on the secondary market
to optimize buying decisions with 40%+ profit margins.

Features:
- Billboard chart data integration
- Spotify artist popularity analysis
- ShowsOnSale.com tour information
- Social media sentiment analysis
- Economic market conditions
- Venue and market tier analysis
- Comprehensive scoring framework
- Purchase recommendations with risk assessment
"""

__version__ = "1.0.0"
__author__ = "Ticket Broker Optimization Team"

# Main application interface
from .main import TicketBrokerOptimizer, run_example_analysis

# Core scoring engine
from .utils.scoring_engine import TicketBrokerScoringEngine

# Data collectors
from .data_collectors.spotify_collector import SpotifyCollector
from .data_collectors.billboard_collector import BillboardCollector
from .data_collectors.showsonsale_collector import ShowsOnSaleCollector

# Models and types
from .models.event_models import (
    RecommendationLevel,
    PurchaseRecommendation,
    MarketAnalysis,
    EventResponse,
    ArtistResponse,
    VenueResponse
)

# Configuration
from .config.settings import settings

__all__ = [
    # Main interface
    'TicketBrokerOptimizer',
    'run_example_analysis',
    
    # Core components
    'TicketBrokerScoringEngine',
    
    # Data collectors
    'SpotifyCollector',
    'BillboardCollector',
    'ShowsOnSaleCollector',
    
    # Models
    'RecommendationLevel',
    'PurchaseRecommendation',
    'MarketAnalysis',
    'EventResponse',
    'ArtistResponse',
    'VenueResponse',
    
    # Configuration
    'settings'
]