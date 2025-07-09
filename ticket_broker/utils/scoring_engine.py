"""
Ticket Broker Scoring Engine - Core Analysis System
Implements the comprehensive scoring framework for event profitability analysis
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from ..config.settings import (
    settings, HIGH_PROFIT_INDICATORS, MEDIUM_PROFIT_INDICATORS, 
    LOW_PROFIT_INDICATORS, RISK_FACTORS, SCORING_WEIGHTS, VENUE_CATEGORIES, MARKET_TIERS
)
from ..models.event_models import RecommendationLevel, MarketAnalysis, PurchaseRecommendation
from ..data_collectors.spotify_collector import SpotifyCollector
from ..data_collectors.billboard_collector import BillboardCollector
from ..data_collectors.showsonsale_collector import ShowsOnSaleCollector


logger = logging.getLogger(__name__)


@dataclass
class ScoringComponents:
    """Data structure for organizing scoring components"""
    high_profit_indicators: List[str]
    medium_profit_indicators: List[str]
    low_profit_indicators: List[str]
    risk_factors: List[str]
    detailed_analysis: Dict[str, Any]
    confidence_multipliers: Dict[str, float]


class TicketBrokerScoringEngine:
    """
    Comprehensive scoring engine for ticket broker investment decisions
    Implements the pre-purchase event evaluation checklist and scoring system
    """
    
    def __init__(self):
        self.spotify_collector = SpotifyCollector()
        self.billboard_collector = BillboardCollector()
        self.showsonsale_collector = ShowsOnSaleCollector()
        
    def analyze_event(self, event_data: Dict[str, Any]) -> PurchaseRecommendation:
        """
        Main entry point for comprehensive event analysis
        Returns complete purchase recommendation with reasoning
        """
        try:
            logger.info(f"Starting analysis for event: {event_data.get('name', 'Unknown')}")
            
            # Gather all data sources
            market_analysis = self._perform_market_analysis(event_data)
            scoring_components = self._calculate_scoring_components(event_data, market_analysis)
            
            # Calculate final score and recommendation
            total_score = self._calculate_total_score(scoring_components)
            recommendation = self._determine_recommendation(total_score, scoring_components)
            confidence_score = self._calculate_confidence_score(scoring_components)
            
            # Generate detailed reasoning
            reasoning = self._generate_reasoning(event_data, scoring_components, market_analysis)
            
            # Calculate financial projections
            financial_analysis = self._analyze_financial_potential(event_data, market_analysis)
            
            return PurchaseRecommendation(
                event_id=event_data.get('id', 0),
                recommendation=recommendation,
                confidence_score=confidence_score,
                total_score=total_score,
                expected_profit_margin=financial_analysis['expected_profit_margin'],
                max_recommended_investment=financial_analysis['max_investment'],
                key_factors=scoring_components.high_profit_indicators + scoring_components.medium_profit_indicators,
                risk_factors=scoring_components.risk_factors,
                reasoning=reasoning,
                action_required_by=self._calculate_action_deadline(event_data)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing event: {e}")
            return self._create_error_recommendation(event_data)
    
    def _perform_market_analysis(self, event_data: Dict[str, Any]) -> MarketAnalysis:
        """Comprehensive market analysis using all data sources"""
        try:
            artist_name = event_data.get('artist_name', '')
            venue_city = event_data.get('venue_city', '')
            venue_name = event_data.get('venue_name', '')
            event_date = event_data.get('event_date')
            
            # Artist Popularity Analysis
            artist_analysis = self._analyze_artist_popularity(artist_name)
            
            # Social Media Buzz Analysis
            social_buzz = self._analyze_social_media_buzz(artist_name)
            
            # Local Market Analysis
            local_market = self._analyze_local_market(venue_city, venue_name)
            
            # Competition Analysis
            competition = self._analyze_competition(event_date, venue_city)
            
            # Economic Conditions
            economic_conditions = self._analyze_economic_conditions(venue_city)
            
            # Venue Premium Analysis
            venue_premium = self._analyze_venue_premium(venue_name, venue_city)
            
            # Timing Analysis
            timing_score = self._analyze_timing_factors(event_date, artist_name)
            
            # Overall demand prediction
            demand_prediction = self._predict_overall_demand(
                artist_analysis, social_buzz, local_market, competition, timing_score
            )
            
            # Purchase limit calculation
            purchase_limit = self._calculate_suggested_purchase_limit(
                demand_prediction, venue_premium, economic_conditions
            )
            
            return MarketAnalysis(
                event_id=event_data.get('id', 0),
                artist_popularity_trend=artist_analysis['trend'],
                social_media_buzz=social_buzz,
                local_market_strength=local_market,
                competition_level=competition,
                economic_conditions=economic_conditions,
                venue_premium=venue_premium,
                timing_score=timing_score,
                overall_demand_prediction=demand_prediction,
                suggested_purchase_limit=purchase_limit,
                risk_assessment=self._assess_overall_risk(competition, economic_conditions, demand_prediction)
            )
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._create_default_market_analysis(event_data)
    
    def _analyze_artist_popularity(self, artist_name: str) -> Dict[str, Any]:
        """Analyze artist popularity trends using Spotify and Billboard data"""
        try:
            analysis = {
                'trend': 'stable',
                'spotify_popularity': 0,
                'chart_momentum': 'stable',
                'recent_releases': False,
                'career_stage': 'established',
                'tour_frequency': 1.0
            }
            
            # Spotify Analysis
            spotify_data = self.spotify_collector.search_artist(artist_name)
            if spotify_data:
                detailed_spotify = self.spotify_collector.get_artist_details(spotify_data['spotify_id'])
                if detailed_spotify:
                    analysis['spotify_popularity'] = detailed_spotify.get('popularity', 0)
                    analysis['recent_releases'] = self.spotify_collector.check_new_releases(
                        spotify_data['spotify_id'], months_back=6
                    )
                    
                    trend_data = self.spotify_collector.analyze_popularity_trend(spotify_data['spotify_id'])
                    if trend_data:
                        analysis['trend'] = trend_data.get('trend', 'stable')
                        analysis['career_stage'] = self._determine_career_stage(detailed_spotify)
            
            # Billboard Analysis
            billboard_data = self.billboard_collector.analyze_chart_momentum(artist_name)
            if billboard_data:
                analysis['chart_momentum'] = billboard_data.get('overall_momentum', 'stable')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing artist popularity for {artist_name}: {e}")
            return {'trend': 'stable', 'spotify_popularity': 50, 'chart_momentum': 'stable'}
    
    def _analyze_social_media_buzz(self, artist_name: str) -> float:
        """Analyze social media buzz and fan excitement (0.0 to 1.0)"""
        try:
            # This would integrate with Twitter API, Instagram, TikTok, etc.
            # For now, return a calculated score based on available data
            
            buzz_score = 0.5  # Default moderate buzz
            
            # Factors that would increase buzz:
            # - Recent tweets mentioning the artist
            # - Instagram engagement rates
            # - TikTok trending sounds
            # - Fan forum activity
            # - Google Trends data
            
            # Billboard chart presence indicates buzz
            billboard_data = self.billboard_collector.get_multiple_chart_positions(artist_name)
            if billboard_data:
                active_charts = len([chart for chart, data in billboard_data.items() if data.get('current_position')])
                buzz_score += min(active_charts * 0.1, 0.3)
            
            # Recent releases boost buzz
            spotify_data = self.spotify_collector.search_artist(artist_name)
            if spotify_data:
                has_recent_release = self.spotify_collector.check_new_releases(spotify_data['spotify_id'])
                if has_recent_release:
                    buzz_score += 0.2
            
            return min(buzz_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing social media buzz for {artist_name}: {e}")
            return 0.5
    
    def _analyze_local_market(self, city: str, venue_name: str) -> float:
        """Analyze local market strength (0.0 to 1.0)"""
        try:
            market_strength = 0.5  # Default moderate strength
            
            # Market tier analysis
            if city:
                city_upper = city.upper()
                if any(tier1_city.upper() in city_upper for tier1_city in MARKET_TIERS['tier_1']):
                    market_strength = 0.9
                elif any(tier2_city.upper() in city_upper for tier2_city in MARKET_TIERS['tier_2']):
                    market_strength = 0.7
                elif any(tier3_city.upper() in city_upper for tier3_city in MARKET_TIERS['tier_3']):
                    market_strength = 0.6
            
            # Venue analysis from ShowsOnSale
            if venue_name:
                venue_info = self.showsonsale_collector.get_venue_information(venue_name, city)
                if venue_info:
                    # Premium venues in good locations boost market strength
                    if venue_info.get('venue_type') in ['arena', 'stadium']:
                        market_strength += 0.1
                    elif venue_info.get('venue_type') in ['theater', 'hall']:
                        market_strength += 0.05
            
            return min(market_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing local market for {city}: {e}")
            return 0.5
    
    def _analyze_competition(self, event_date: str, city: str) -> float:
        """Analyze competition level on event date (0.0 = no competition, 1.0 = high competition)"""
        try:
            if not event_date:
                return 0.3  # Default low competition
            
            competition_level = 0.0
            
            # Check for competing events using ShowsOnSale data
            # This would look for other major events on the same weekend
            event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            weekend_start = event_datetime - timedelta(days=1)
            weekend_end = event_datetime + timedelta(days=1)
            
            # Get upcoming events in the same city
            market_trends = self.showsonsale_collector.get_market_trends(city=city)
            upcoming_events = market_trends.get('upcoming_major_events', [])
            
            # High competition if multiple major events
            if len(upcoming_events) > 3:
                competition_level += 0.4
            elif len(upcoming_events) > 1:
                competition_level += 0.2
            
            # Holiday weekends typically have more competition
            if self._is_holiday_weekend(event_datetime):
                competition_level += 0.2
            
            return min(competition_level, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing competition: {e}")
            return 0.3
    
    def _analyze_economic_conditions(self, city: str) -> float:
        """Analyze local economic conditions (0.0 = poor, 1.0 = excellent)"""
        try:
            # This would integrate with economic APIs (FRED, Bureau of Labor Statistics)
            # For now, provide reasonable defaults based on market tiers
            
            economic_score = 0.7  # Default good conditions
            
            if city:
                city_upper = city.upper()
                if any(tier1_city.upper() in city_upper for tier1_city in MARKET_TIERS['tier_1']):
                    economic_score = 0.85  # Tier 1 cities typically strong
                elif any(tier2_city.upper() in city_upper for tier2_city in MARKET_TIERS['tier_2']):
                    economic_score = 0.75  # Tier 2 cities generally good
                elif any(tier3_city.upper() in city_upper for tier3_city in MARKET_TIERS['tier_3']):
                    economic_score = 0.65  # Tier 3 cities more variable
            
            return economic_score
            
        except Exception as e:
            logger.error(f"Error analyzing economic conditions for {city}: {e}")
            return 0.7
    
    def _analyze_venue_premium(self, venue_name: str, city: str) -> float:
        """Calculate venue premium multiplier"""
        try:
            premium = 1.0  # Default no premium
            
            if venue_name:
                venue_info = self.showsonsale_collector.get_venue_information(venue_name, city)
                if venue_info:
                    capacity = venue_info.get('capacity', 10000)
                    venue_type = venue_info.get('venue_type', 'unknown')
                    
                    # Apply venue category multipliers
                    for category, details in VENUE_CATEGORIES.items():
                        if capacity <= details['max_capacity']:
                            premium = details['multiplier']
                            break
                    
                    # Premium venues get additional boost
                    if venue_type in ['arena', 'stadium'] and capacity < 20000:
                        premium += 0.1
                    elif venue_type in ['theater', 'hall'] and capacity < 5000:
                        premium += 0.2
            
            return premium
            
        except Exception as e:
            logger.error(f"Error analyzing venue premium for {venue_name}: {e}")
            return 1.0
    
    def _analyze_timing_factors(self, event_date: str, artist_name: str) -> float:
        """Analyze timing factors (0.0 to 1.0)"""
        try:
            if not event_date:
                return 0.5
            
            timing_score = 0.5
            event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            
            # Weekend shows typically perform better
            if event_datetime.weekday() in [4, 5, 6]:  # Friday, Saturday, Sunday
                timing_score += 0.2
            
            # Holiday proximity
            if self._is_holiday_weekend(event_datetime):
                timing_score += 0.1
            
            # Summer tour season (better for outdoor venues)
            if event_datetime.month in [6, 7, 8, 9]:
                timing_score += 0.1
            
            # Award season boost (Grammy season, etc.)
            if event_datetime.month in [1, 2, 3]:  # Award season
                award_impact = self.billboard_collector.check_award_season_impact(artist_name)
                if award_impact.get('award_season_boost'):
                    timing_score += 0.2
            
            return min(timing_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing timing factors: {e}")
            return 0.5
    
    def _calculate_scoring_components(self, event_data: Dict[str, Any], market_analysis: MarketAnalysis) -> ScoringComponents:
        """Calculate all scoring components based on the framework"""
        try:
            high_profit = []
            medium_profit = []
            low_profit = []
            risk_factors = []
            detailed_analysis = {}
            confidence_multipliers = {}
            
            # Artist/Team Performance Analysis
            artist_analysis = self._score_artist_performance(event_data, market_analysis)
            high_profit.extend(artist_analysis['high_profit'])
            medium_profit.extend(artist_analysis['medium_profit'])
            low_profit.extend(artist_analysis['low_profit'])
            risk_factors.extend(artist_analysis['risk_factors'])
            detailed_analysis['artist_performance'] = artist_analysis['details']
            confidence_multipliers.update(artist_analysis['confidence'])
            
            # Venue & Market Factors
            venue_analysis = self._score_venue_market_factors(event_data, market_analysis)
            high_profit.extend(venue_analysis['high_profit'])
            medium_profit.extend(venue_analysis['medium_profit'])
            low_profit.extend(venue_analysis['low_profit'])
            risk_factors.extend(venue_analysis['risk_factors'])
            detailed_analysis['venue_market'] = venue_analysis['details']
            confidence_multipliers.update(venue_analysis['confidence'])
            
            # Timing & Demand Indicators
            timing_analysis = self._score_timing_demand(event_data, market_analysis)
            high_profit.extend(timing_analysis['high_profit'])
            medium_profit.extend(timing_analysis['medium_profit'])
            low_profit.extend(timing_analysis['low_profit'])
            risk_factors.extend(timing_analysis['risk_factors'])
            detailed_analysis['timing_demand'] = timing_analysis['details']
            confidence_multipliers.update(timing_analysis['confidence'])
            
            # Pricing Intelligence
            pricing_analysis = self._score_pricing_intelligence(event_data, market_analysis)
            high_profit.extend(pricing_analysis['high_profit'])
            medium_profit.extend(pricing_analysis['medium_profit'])
            low_profit.extend(pricing_analysis['low_profit'])
            risk_factors.extend(pricing_analysis['risk_factors'])
            detailed_analysis['pricing'] = pricing_analysis['details']
            confidence_multipliers.update(pricing_analysis['confidence'])
            
            return ScoringComponents(
                high_profit_indicators=high_profit,
                medium_profit_indicators=medium_profit,
                low_profit_indicators=low_profit,
                risk_factors=risk_factors,
                detailed_analysis=detailed_analysis,
                confidence_multipliers=confidence_multipliers
            )
            
        except Exception as e:
            logger.error(f"Error calculating scoring components: {e}")
            return ScoringComponents([], [], [], [], {}, {})
    
    def _score_artist_performance(self, event_data: Dict[str, Any], market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """Score artist/team performance indicators"""
        analysis = {
            'high_profit': [],
            'medium_profit': [],
            'low_profit': [],
            'risk_factors': [],
            'details': {},
            'confidence': {}
        }
        
        try:
            artist_name = event_data.get('artist_name', '')
            
            # Popularity trends
            if market_analysis.artist_popularity_trend == 'rising':
                analysis['medium_profit'].append('rising_artist_popularity')
                analysis['confidence']['artist_trend'] = 0.8
            elif market_analysis.artist_popularity_trend == 'declining':
                analysis['risk_factors'].append('declining_popularity')
                analysis['confidence']['artist_trend'] = 0.9
            else:
                analysis['low_profit'].append('stable_artist_popularity')
                analysis['confidence']['artist_trend'] = 0.6
            
            # Tour frequency analysis
            spotify_data = self.spotify_collector.search_artist(artist_name)
            if spotify_data:
                detailed_data = self.spotify_collector.get_artist_details(spotify_data['spotify_id'])
                if detailed_data:
                    # Check for first tour in 2+ years
                    # This would require historical tour data analysis
                    pass
            
            # Recent album releases
            if spotify_data:
                has_recent_album = self.spotify_collector.check_new_releases(spotify_data['spotify_id'], 6)
                if has_recent_album:
                    analysis['medium_profit'].append('new_album_within_6_months')
                    analysis['confidence']['recent_release'] = 0.7
            
            # Chart performance
            chart_data = self.billboard_collector.analyze_chart_momentum(artist_name)
            if chart_data:
                if chart_data.get('market_presence') == 'high':
                    analysis['medium_profit'].append('strong_chart_presence')
                    analysis['confidence']['chart_performance'] = 0.8
                elif chart_data.get('overall_momentum') == 'rising':
                    analysis['medium_profit'].append('rising_chart_momentum')
                    analysis['confidence']['chart_performance'] = 0.7
            
            analysis['details'] = {
                'popularity_trend': market_analysis.artist_popularity_trend,
                'social_buzz_score': market_analysis.social_media_buzz,
                'chart_momentum': chart_data.get('overall_momentum', 'unknown') if chart_data else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error scoring artist performance: {e}")
        
        return analysis
    
    def _score_venue_market_factors(self, event_data: Dict[str, Any], market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """Score venue and market factors"""
        analysis = {
            'high_profit': [],
            'medium_profit': [],
            'low_profit': [],
            'risk_factors': [],
            'details': {},
            'confidence': {}
        }
        
        try:
            venue_name = event_data.get('venue_name', '')
            venue_city = event_data.get('venue_city', '')
            
            # Venue size analysis
            venue_info = self.showsonsale_collector.get_venue_information(venue_name, venue_city)
            if venue_info:
                capacity = venue_info.get('capacity', 10000)
                
                if capacity < 5000:
                    analysis['high_profit'].append('venue_under_5000_capacity')
                    analysis['confidence']['venue_size'] = 0.9
                elif capacity > 20000:
                    analysis['risk_factors'].append('large_venue_20k_plus')
                    analysis['confidence']['venue_size'] = 0.8
                else:
                    analysis['medium_profit'].append('mid_tier_venue')
                    analysis['confidence']['venue_size'] = 0.6
                
                # Premium venue check
                venue_type = venue_info.get('venue_type', '')
                if venue_type in ['arena', 'theater'] and capacity < 15000:
                    analysis['medium_profit'].append('premium_venue')
                    analysis['confidence']['venue_quality'] = 0.7
            
            # Market strength
            if market_analysis.local_market_strength > 0.8:
                analysis['medium_profit'].append('strong_local_fanbase')
                analysis['confidence']['market_strength'] = 0.8
            elif market_analysis.local_market_strength < 0.4:
                analysis['risk_factors'].append('weak_local_market')
                analysis['confidence']['market_strength'] = 0.7
            
            # Competition level
            if market_analysis.competition_level > 0.7:
                analysis['risk_factors'].append('competing_major_events')
                analysis['confidence']['competition'] = 0.8
            
            analysis['details'] = {
                'venue_capacity': venue_info.get('capacity') if venue_info else None,
                'venue_type': venue_info.get('venue_type') if venue_info else None,
                'market_strength': market_analysis.local_market_strength,
                'competition_level': market_analysis.competition_level
            }
            
        except Exception as e:
            logger.error(f"Error scoring venue/market factors: {e}")
        
        return analysis
    
    def _score_timing_demand(self, event_data: Dict[str, Any], market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """Score timing and demand indicators"""
        analysis = {
            'high_profit': [],
            'medium_profit': [],
            'low_profit': [],
            'risk_factors': [],
            'details': {},
            'confidence': {}
        }
        
        try:
            event_date = event_data.get('event_date')
            is_farewell = event_data.get('is_farewell_tour', False)
            
            # Farewell tour check
            if is_farewell:
                analysis['high_profit'].append('farewell_retirement_tour')
                analysis['confidence']['farewell_tour'] = 0.95
            
            # Weekend performance
            if event_date:
                event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                if event_datetime.weekday() in [4, 5, 6]:  # Friday, Saturday, Sunday
                    analysis['medium_profit'].append('weekend_performance')
                    analysis['confidence']['weekend'] = 0.7
                
                # Holiday/special date
                if self._is_holiday_weekend(event_datetime):
                    analysis['medium_profit'].append('holiday_special_date')
                    analysis['confidence']['holiday'] = 0.6
            
            # Sellout speed (if available)
            sellout_time = event_data.get('sellout_time_minutes')
            if sellout_time and sellout_time < 60:
                analysis['high_profit'].append('sold_out_under_1_hour')
                analysis['confidence']['sellout_speed'] = 0.9
            
            # Social media buzz
            if market_analysis.social_media_buzz > 0.8:
                analysis['medium_profit'].append('high_social_buzz')
                analysis['confidence']['social_buzz'] = 0.7
            elif market_analysis.social_media_buzz > 0.6:
                analysis['low_profit'].append('moderate_social_buzz')
                analysis['confidence']['social_buzz'] = 0.5
            
            analysis['details'] = {
                'timing_score': market_analysis.timing_score,
                'social_buzz': market_analysis.social_media_buzz,
                'is_weekend': event_datetime.weekday() in [4, 5, 6] if event_date else False,
                'is_farewell_tour': is_farewell
            }
            
        except Exception as e:
            logger.error(f"Error scoring timing/demand: {e}")
        
        return analysis
    
    def _score_pricing_intelligence(self, event_data: Dict[str, Any], market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """Score pricing intelligence factors"""
        analysis = {
            'high_profit': [],
            'medium_profit': [],
            'low_profit': [],
            'risk_factors': [],
            'details': {},
            'confidence': {}
        }
        
        try:
            face_value_min = event_data.get('face_value_min', 0)
            face_value_max = event_data.get('face_value_max', 0)
            estimated_resale_min = event_data.get('estimated_resale_min', 0)
            estimated_resale_max = event_data.get('estimated_resale_max', 0)
            
            # Face value analysis
            if face_value_max and face_value_max < 150:  # Reasonable face values
                analysis['low_profit'].append('standard_ticket_prices')
                analysis['confidence']['face_value'] = 0.6
            elif face_value_max and face_value_max > 300:  # High face values
                analysis['risk_factors'].append('high_face_values')
                analysis['confidence']['face_value'] = 0.7
            
            # Resale potential
            if estimated_resale_min and face_value_min:
                potential_margin = (estimated_resale_min / face_value_min) - 1
                if potential_margin > 0.6:  # Over 60% markup potential
                    analysis['high_profit'].append('high_resale_potential')
                    analysis['confidence']['resale_potential'] = 0.8
                elif potential_margin > 0.3:  # Over 30% markup potential
                    analysis['medium_profit'].append('moderate_resale_potential')
                    analysis['confidence']['resale_potential'] = 0.6
                elif potential_margin < 0.1:  # Low markup potential
                    analysis['risk_factors'].append('low_resale_potential')
                    analysis['confidence']['resale_potential'] = 0.7
            
            # Ticketmaster event premium
            is_ticketmaster = event_data.get('is_ticketmaster', False)
            if is_ticketmaster:
                analysis['low_profit'].append('ticketmaster_event')
                analysis['confidence']['ticketmaster'] = 0.5
            
            analysis['details'] = {
                'face_value_range': [face_value_min, face_value_max],
                'estimated_resale_range': [estimated_resale_min, estimated_resale_max],
                'potential_margin': potential_margin if 'potential_margin' in locals() else None,
                'is_ticketmaster': is_ticketmaster
            }
            
        except Exception as e:
            logger.error(f"Error scoring pricing intelligence: {e}")
        
        return analysis
    
    def _calculate_total_score(self, components: ScoringComponents) -> int:
        """Calculate total score based on framework weights"""
        try:
            score = 0
            
            # Apply scoring weights
            score += len(components.high_profit_indicators) * SCORING_WEIGHTS['high_profit_indicators']
            score += len(components.medium_profit_indicators) * SCORING_WEIGHTS['medium_profit_indicators']
            score += len(components.low_profit_indicators) * SCORING_WEIGHTS['low_profit_indicators']
            score += len(components.risk_factors) * SCORING_WEIGHTS['risk_factors']
            
            return max(score, 0)  # Ensure non-negative score
            
        except Exception as e:
            logger.error(f"Error calculating total score: {e}")
            return 0
    
    def _determine_recommendation(self, total_score: int, components: ScoringComponents) -> RecommendationLevel:
        """Determine recommendation level based on score and risk factors"""
        try:
            # Check for high-risk scenarios first
            critical_risks = ['declining_popularity', 'economic_downturn', 'competing_major_events']
            if any(risk in components.risk_factors for risk in critical_risks):
                if total_score < settings.moderate_profit_score_threshold:
                    return RecommendationLevel.HIGH_RISK
            
            # Apply standard thresholds
            if total_score >= settings.high_profit_score_threshold:
                return RecommendationLevel.STRONG_BUY
            elif total_score >= settings.moderate_profit_score_threshold:
                return RecommendationLevel.SELECTIVE_BUY
            elif total_score >= settings.low_profit_score_threshold:
                return RecommendationLevel.AVOID
            else:
                return RecommendationLevel.HIGH_RISK
                
        except Exception as e:
            logger.error(f"Error determining recommendation: {e}")
            return RecommendationLevel.HIGH_RISK
    
    def _calculate_confidence_score(self, components: ScoringComponents) -> float:
        """Calculate confidence score based on data quality and certainty"""
        try:
            if not components.confidence_multipliers:
                return 0.5
            
            # Average confidence across all factors
            confidence_values = list(components.confidence_multipliers.values())
            base_confidence = sum(confidence_values) / len(confidence_values)
            
            # Adjust based on data completeness
            data_completeness = len(components.confidence_multipliers) / 10  # Expect ~10 factors
            completeness_factor = min(data_completeness, 1.0)
            
            return min(base_confidence * completeness_factor, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _analyze_financial_potential(self, event_data: Dict[str, Any], market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """Analyze financial potential and calculate investment limits"""
        try:
            face_value_avg = ((event_data.get('face_value_min', 0) + event_data.get('face_value_max', 0)) / 2) or 100
            estimated_resale_avg = ((event_data.get('estimated_resale_min', 0) + event_data.get('estimated_resale_max', 0)) / 2) or 150
            
            # Calculate expected profit margin
            if face_value_avg > 0:
                expected_margin = (estimated_resale_avg / face_value_avg) - 1
            else:
                expected_margin = 0.2  # Default 20% if no data
            
            # Apply market factors
            market_multiplier = (
                market_analysis.venue_premium * 
                (1 + market_analysis.local_market_strength * 0.2) *
                (1 + market_analysis.overall_demand_prediction * 0.3)
            )
            
            adjusted_margin = expected_margin * market_multiplier
            
            # Calculate maximum investment
            base_investment = min(settings.max_investment_per_event, market_analysis.suggested_purchase_limit)
            
            # Adjust based on confidence and risk
            risk_adjustment = 1.0
            if market_analysis.risk_assessment == 'high':
                risk_adjustment = 0.5
            elif market_analysis.risk_assessment == 'medium':
                risk_adjustment = 0.75
            
            max_investment = base_investment * risk_adjustment
            
            return {
                'expected_profit_margin': max(adjusted_margin, 0.0),
                'max_investment': max_investment,
                'market_multiplier': market_multiplier,
                'risk_adjustment': risk_adjustment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing financial potential: {e}")
            return {'expected_profit_margin': 0.2, 'max_investment': 1000.0}
    
    def _generate_reasoning(self, event_data: Dict[str, Any], components: ScoringComponents, market_analysis: MarketAnalysis) -> str:
        """Generate detailed reasoning for the recommendation"""
        try:
            reasoning_parts = []
            
            # Event overview
            artist_name = event_data.get('artist_name', 'Unknown Artist')
            venue_name = event_data.get('venue_name', 'Unknown Venue')
            venue_city = event_data.get('venue_city', 'Unknown City')
            
            reasoning_parts.append(f"Analysis for {artist_name} at {venue_name} in {venue_city}:")
            
            # Key strengths
            if components.high_profit_indicators:
                reasoning_parts.append(f"Key strengths: {', '.join(components.high_profit_indicators)}")
            
            if components.medium_profit_indicators:
                reasoning_parts.append(f"Supporting factors: {', '.join(components.medium_profit_indicators)}")
            
            # Risk factors
            if components.risk_factors:
                reasoning_parts.append(f"Risk factors: {', '.join(components.risk_factors)}")
            
            # Market analysis summary
            reasoning_parts.append(
                f"Market analysis shows {market_analysis.artist_popularity_trend} artist trend, "
                f"{market_analysis.social_media_buzz:.1%} social buzz, "
                f"and {market_analysis.local_market_strength:.1%} local market strength."
            )
            
            # Financial summary
            if 'pricing' in components.detailed_analysis:
                pricing_details = components.detailed_analysis['pricing']['details']
                if pricing_details.get('potential_margin'):
                    reasoning_parts.append(
                        f"Expected profit margin of {pricing_details['potential_margin']:.1%} "
                        f"based on face value vs. estimated resale prices."
                    )
            
            return " ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Analysis completed with limited data. Recommend manual review."
    
    # Helper methods
    def _calculate_action_deadline(self, event_data: Dict[str, Any]) -> Optional[datetime]:
        """Calculate when action is required by"""
        try:
            on_sale_date = event_data.get('on_sale_date')
            if on_sale_date:
                onsale_datetime = datetime.fromisoformat(on_sale_date.replace('Z', '+00:00'))
                return onsale_datetime - timedelta(hours=2)  # 2 hours before on-sale
            return None
        except:
            return None
    
    def _is_holiday_weekend(self, event_date: datetime) -> bool:
        """Check if event is on a holiday weekend"""
        # Simplified holiday check - would expand with actual holiday calendar
        holiday_months = [11, 12, 1, 7]  # Thanksgiving, Christmas, New Year, July 4th
        return event_date.month in holiday_months
    
    def _determine_career_stage(self, spotify_data: Dict[str, Any]) -> str:
        """Determine artist's career stage"""
        popularity = spotify_data.get('popularity', 50)
        total_albums = spotify_data.get('total_albums', 1)
        
        if popularity > 80 and total_albums > 5:
            return 'established'
        elif popularity > 60 or total_albums < 3:
            return 'rising'
        else:
            return 'declining'
    
    def _predict_overall_demand(self, artist_analysis: Dict, social_buzz: float, 
                              local_market: float, competition: float, timing: float) -> float:
        """Predict overall demand score"""
        try:
            # Weight different factors
            weights = {
                'artist_trend': 0.3,
                'social_buzz': 0.25,
                'local_market': 0.2,
                'timing': 0.15,
                'competition': -0.1  # Competition reduces demand
            }
            
            artist_score = 0.7 if artist_analysis.get('trend') == 'rising' else 0.5
            
            demand_score = (
                weights['artist_trend'] * artist_score +
                weights['social_buzz'] * social_buzz +
                weights['local_market'] * local_market +
                weights['timing'] * timing +
                weights['competition'] * competition
            )
            
            return max(min(demand_score, 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error predicting demand: {e}")
            return 0.5
    
    def _calculate_suggested_purchase_limit(self, demand_prediction: float, 
                                          venue_premium: float, economic_conditions: float) -> float:
        """Calculate suggested purchase limit"""
        try:
            base_limit = settings.max_investment_per_event
            
            # Adjust based on demand and market conditions
            market_multiplier = demand_prediction * venue_premium * economic_conditions
            
            suggested_limit = base_limit * market_multiplier
            
            return min(suggested_limit, settings.max_total_inventory_value * 0.2)  # Max 20% of total inventory
            
        except Exception as e:
            logger.error(f"Error calculating purchase limit: {e}")
            return settings.max_investment_per_event * 0.5
    
    def _assess_overall_risk(self, competition: float, economic_conditions: float, demand_prediction: float) -> str:
        """Assess overall risk level"""
        try:
            risk_score = competition + (1 - economic_conditions) + (1 - demand_prediction)
            
            if risk_score > 1.5:
                return 'high'
            elif risk_score > 0.8:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return 'medium'
    
    def _create_error_recommendation(self, event_data: Dict[str, Any]) -> PurchaseRecommendation:
        """Create error recommendation when analysis fails"""
        return PurchaseRecommendation(
            event_id=event_data.get('id', 0),
            recommendation=RecommendationLevel.HIGH_RISK,
            confidence_score=0.1,
            total_score=0,
            expected_profit_margin=0.0,
            max_recommended_investment=0.0,
            key_factors=[],
            risk_factors=['analysis_failed'],
            reasoning="Analysis failed due to technical error. Manual review required.",
            action_required_by=None
        )
    
    def _create_default_market_analysis(self, event_data: Dict[str, Any]) -> MarketAnalysis:
        """Create default market analysis when data collection fails"""
        return MarketAnalysis(
            event_id=event_data.get('id', 0),
            artist_popularity_trend='stable',
            social_media_buzz=0.5,
            local_market_strength=0.5,
            competition_level=0.5,
            economic_conditions=0.7,
            venue_premium=1.0,
            timing_score=0.5,
            overall_demand_prediction=0.5,
            suggested_purchase_limit=settings.max_investment_per_event * 0.5,
            risk_assessment='medium'
        )