"""
Data Collectors Package

Contains modules for collecting data from various sources:
- Spotify API for artist popularity and streaming data
- Billboard charts for music industry trends
- ShowsOnSale.com for tour information and on-sale dates
- Social media APIs for buzz and sentiment analysis
- Economic data APIs for market conditions
"""

from .spotify_collector import SpotifyCollector
from .billboard_collector import BillboardCollector
from .showsonsale_collector import ShowsOnSaleCollector

__all__ = [
    'SpotifyCollector',
    'BillboardCollector', 
    'ShowsOnSaleCollector'
]