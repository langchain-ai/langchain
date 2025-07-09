"""
Ticket Broker Optimization System - Main Application Interface
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import pandas as pd
from pathlib import Path

from .config.settings import settings
from .utils.scoring_engine import TicketBrokerScoringEngine
from .data_collectors.spotify_collector import SpotifyCollector
from .data_collectors.billboard_collector import BillboardCollector
from .data_collectors.showsonsale_collector import ShowsOnSaleCollector
from .models.event_models import (
    EventCreate, EventResponse, RecommendationLevel, 
    PurchaseRecommendation, MarketAnalysis
)


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TicketBrokerOptimizer:
    """
    Main application class for ticket broker optimization
    Provides high-level interface for event analysis and decision making
    """
    
    def __init__(self):
        self.scoring_engine = TicketBrokerScoringEngine()
        self.spotify_collector = SpotifyCollector()
        self.billboard_collector = BillboardCollector()
        self.showsonsale_collector = ShowsOnSaleCollector()
        
        logger.info("Ticket Broker Optimizer initialized")
    
    def analyze_event(self, event_data: Dict[str, Any]) -> PurchaseRecommendation:
        """
        Analyze a single event and return purchase recommendation
        
        Args:
            event_data: Dictionary containing event information
                Required fields: artist_name, venue_name, venue_city, event_date
                Optional fields: face_value_min, face_value_max, on_sale_date, etc.
        
        Returns:
            PurchaseRecommendation with complete analysis
        """
        try:
            logger.info(f"Analyzing event: {event_data.get('artist_name')} at {event_data.get('venue_name')}")
            
            # Validate required fields
            required_fields = ['artist_name', 'venue_name', 'venue_city', 'event_date']
            missing_fields = [field for field in required_fields if not event_data.get(field)]
            
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Perform comprehensive analysis
            recommendation = self.scoring_engine.analyze_event(event_data)
            
            logger.info(f"Analysis complete. Recommendation: {recommendation.recommendation.value}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error analyzing event: {e}")
            raise
    
    def analyze_multiple_events(self, events_data: List[Dict[str, Any]]) -> List[PurchaseRecommendation]:
        """
        Analyze multiple events and return ranked recommendations
        
        Args:
            events_data: List of event dictionaries
        
        Returns:
            List of PurchaseRecommendations sorted by priority
        """
        try:
            logger.info(f"Analyzing {len(events_data)} events")
            
            recommendations = []
            
            for i, event_data in enumerate(events_data):
                try:
                    logger.info(f"Processing event {i+1}/{len(events_data)}")
                    recommendation = self.analyze_event(event_data)
                    recommendations.append(recommendation)
                except Exception as e:
                    logger.error(f"Error analyzing event {i+1}: {e}")
                    # Continue with other events
                    continue
            
            # Sort by recommendation priority and score
            priority_order = {
                RecommendationLevel.STRONG_BUY: 4,
                RecommendationLevel.SELECTIVE_BUY: 3,
                RecommendationLevel.AVOID: 2,
                RecommendationLevel.HIGH_RISK: 1
            }
            
            recommendations.sort(
                key=lambda x: (priority_order.get(x.recommendation, 0), x.total_score, x.confidence_score),
                reverse=True
            )
            
            logger.info(f"Completed analysis of {len(recommendations)} events")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing multiple events: {e}")
            raise
    
    def search_and_analyze_artist_tours(self, artist_name: str) -> List[PurchaseRecommendation]:
        """
        Search for an artist's upcoming tours and analyze all events
        
        Args:
            artist_name: Name of the artist to search for
        
        Returns:
            List of PurchaseRecommendations for all found events
        """
        try:
            logger.info(f"Searching and analyzing tours for: {artist_name}")
            
            # Search for upcoming shows
            shows = self.showsonsale_collector.search_artist_tours(artist_name)
            
            if not shows:
                logger.warning(f"No upcoming shows found for {artist_name}")
                return []
            
            logger.info(f"Found {len(shows)} upcoming shows for {artist_name}")
            
            # Convert show data to event format and analyze
            events_data = []
            for show in shows:
                event_data = self._convert_show_to_event_data(show)
                events_data.append(event_data)
            
            recommendations = self.analyze_multiple_events(events_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error searching and analyzing tours for {artist_name}: {e}")
            raise
    
    def get_upcoming_onsales_analysis(self, days_ahead: int = 7) -> List[PurchaseRecommendation]:
        """
        Get and analyze upcoming on-sale events
        
        Args:
            days_ahead: Number of days to look ahead for on-sales
        
        Returns:
            List of PurchaseRecommendations for upcoming on-sales
        """
        try:
            logger.info(f"Analyzing upcoming on-sales for next {days_ahead} days")
            
            # Get upcoming on-sales
            onsales = self.showsonsale_collector.get_upcoming_onsales(days_ahead)
            
            if not onsales:
                logger.info("No upcoming on-sales found")
                return []
            
            logger.info(f"Found {len(onsales)} upcoming on-sales")
            
            # Convert to event format and analyze
            events_data = []
            for onsale in onsales:
                event_data = self._convert_onsale_to_event_data(onsale)
                events_data.append(event_data)
            
            recommendations = self.analyze_multiple_events(events_data)
            
            # Filter to only strong buy and selective buy recommendations
            priority_recommendations = [
                rec for rec in recommendations 
                if rec.recommendation in [RecommendationLevel.STRONG_BUY, RecommendationLevel.SELECTIVE_BUY]
            ]
            
            logger.info(f"Found {len(priority_recommendations)} priority recommendations from upcoming on-sales")
            
            return priority_recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing upcoming on-sales: {e}")
            raise
    
    def generate_market_report(self, city: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive market report for a city or overall market
        
        Args:
            city: City to analyze (optional, analyzes overall market if None)
        
        Returns:
            Dictionary containing market analysis report
        """
        try:
            logger.info(f"Generating market report for: {city or 'overall market'}")
            
            report = {
                'report_date': datetime.utcnow().isoformat(),
                'city': city,
                'market_trends': {},
                'hot_artists': [],
                'upcoming_opportunities': [],
                'risk_factors': [],
                'recommendations': []
            }
            
            # Get market trends
            market_trends = self.showsonsale_collector.get_market_trends(city=city)
            report['market_trends'] = market_trends
            
            # Get Billboard industry trends
            industry_trends = self.billboard_collector.get_industry_trends()
            report['hot_artists'] = industry_trends.get('hot_artists', [])
            
            # Get upcoming on-sales for opportunities
            upcoming_onsales = self.get_upcoming_onsales_analysis(days_ahead=30)
            report['upcoming_opportunities'] = [
                {
                    'artist': rec.key_factors,
                    'recommendation': rec.recommendation.value,
                    'score': rec.total_score,
                    'expected_margin': rec.expected_profit_margin
                }
                for rec in upcoming_onsales[:10]  # Top 10 opportunities
            ]
            
            # Generate recommendations
            report['recommendations'] = self._generate_market_recommendations(report)
            
            logger.info("Market report generated successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating market report: {e}")
            raise
    
    def export_analysis_report(self, recommendations: List[PurchaseRecommendation], 
                             filename: str = None) -> str:
        """
        Export analysis results to a detailed report file
        
        Args:
            recommendations: List of recommendations to export
            filename: Output filename (optional, auto-generated if None)
        
        Returns:
            Path to the generated report file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ticket_broker_analysis_{timestamp}.json"
            
            # Ensure output directory exists
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            
            filepath = output_dir / filename
            
            # Convert recommendations to serializable format
            report_data = {
                'generated_at': datetime.utcnow().isoformat(),
                'total_events_analyzed': len(recommendations),
                'summary': self._generate_summary_stats(recommendations),
                'recommendations': []
            }
            
            for rec in recommendations:
                rec_dict = {
                    'event_id': rec.event_id,
                    'recommendation': rec.recommendation.value,
                    'confidence_score': rec.confidence_score,
                    'total_score': rec.total_score,
                    'expected_profit_margin': rec.expected_profit_margin,
                    'max_recommended_investment': rec.max_recommended_investment,
                    'key_factors': rec.key_factors,
                    'risk_factors': rec.risk_factors,
                    'reasoning': rec.reasoning,
                    'action_required_by': rec.action_required_by.isoformat() if rec.action_required_by else None
                }
                report_data['recommendations'].append(rec_dict)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Analysis report exported to: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting analysis report: {e}")
            raise
    
    def quick_decision_framework(self, artist_name: str, venue_name: str, 
                                venue_city: str, event_date: str) -> Dict[str, Any]:
        """
        Implement the 60-second gut check and quick decision framework
        
        Args:
            artist_name: Name of the artist
            venue_name: Name of the venue
            venue_city: City where venue is located
            event_date: Date of the event (ISO format)
        
        Returns:
            Quick decision analysis with go/no-go recommendation
        """
        try:
            logger.info(f"Quick decision analysis for: {artist_name}")
            
            quick_analysis = {
                'artist_name': artist_name,
                'venue_name': venue_name,
                'venue_city': venue_city,
                'event_date': event_date,
                'analysis_time': datetime.utcnow().isoformat(),
                'gut_check': {},
                'quick_score': 0,
                'decision': 'investigate_further',
                'reasoning': '',
                'next_steps': []
            }
            
            # 60-Second Gut Check
            gut_check_score = 0
            
            # Check if artist is having a moment (Billboard data)
            chart_momentum = self.billboard_collector.analyze_chart_momentum(artist_name)
            if chart_momentum.get('overall_momentum') == 'rising':
                gut_check_score += 2
                quick_analysis['gut_check']['chart_momentum'] = 'rising'
            elif chart_momentum.get('market_presence') == 'high':
                gut_check_score += 1
                quick_analysis['gut_check']['chart_momentum'] = 'strong_presence'
            
            # Check for recent releases (Spotify data)
            spotify_data = self.spotify_collector.search_artist(artist_name)
            if spotify_data:
                has_recent_release = self.spotify_collector.check_new_releases(spotify_data['spotify_id'])
                if has_recent_release:
                    gut_check_score += 1
                    quick_analysis['gut_check']['recent_release'] = True
            
            # Market tier check
            city_upper = venue_city.upper()
            from .config.settings import MARKET_TIERS
            
            if any(tier1_city.upper() in city_upper for tier1_city in MARKET_TIERS['tier_1']):
                gut_check_score += 2
                quick_analysis['gut_check']['market_tier'] = 'tier_1'
            elif any(tier2_city.upper() in city_upper for tier2_city in MARKET_TIERS['tier_2']):
                gut_check_score += 1
                quick_analysis['gut_check']['market_tier'] = 'tier_2'
            
            quick_analysis['quick_score'] = gut_check_score
            
            # Make quick decision
            if gut_check_score >= 4:
                quick_analysis['decision'] = 'proceed_with_analysis'
                quick_analysis['reasoning'] = 'Strong indicators present. Recommend full analysis.'
                quick_analysis['next_steps'] = ['Perform full 15-minute analysis', 'Check venue capacity', 'Verify on-sale date']
            elif gut_check_score >= 2:
                quick_analysis['decision'] = 'investigate_further'
                quick_analysis['reasoning'] = 'Some positive indicators. Worth deeper investigation.'
                quick_analysis['next_steps'] = ['Check social media buzz', 'Verify tour status', 'Compare to similar events']
            else:
                quick_analysis['decision'] = 'likely_pass'
                quick_analysis['reasoning'] = 'Limited positive indicators. Likely not profitable.'
                quick_analysis['next_steps'] = ['Monitor for changes', 'Check for special circumstances']
            
            logger.info(f"Quick decision: {quick_analysis['decision']} (score: {gut_check_score})")
            
            return quick_analysis
            
        except Exception as e:
            logger.error(f"Error in quick decision framework: {e}")
            raise
    
    def monitor_portfolio_performance(self) -> Dict[str, Any]:
        """
        Monitor performance of current ticket portfolio
        Note: This would require integration with actual purchase/sales tracking
        """
        logger.info("Portfolio monitoring feature - requires purchase tracking integration")
        
        return {
            'status': 'not_implemented',
            'message': 'Portfolio monitoring requires integration with purchase and sales tracking systems'
        }
    
    # Helper methods
    def _convert_show_to_event_data(self, show: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ShowsOnSale show data to event analysis format"""
        return {
            'id': hash(f"{show.get('artist_name', '')}{show.get('venue_name', '')}{show.get('event_date', '')}"),
            'artist_name': show.get('artist_name', ''),
            'venue_name': show.get('venue_name', ''),
            'venue_city': show.get('venue_city', ''),
            'venue_state': show.get('venue_state', ''),
            'event_date': show.get('event_date', ''),
            'on_sale_date': show.get('on_sale_date', ''),
            'tour_name': show.get('tour_name', ''),
            'face_value_min': show.get('price_range', {}).get('min') if show.get('price_range') else None,
            'face_value_max': show.get('price_range', {}).get('max') if show.get('price_range') else None,
            'is_ticketmaster': 'ticketmaster' in show.get('ticket_url', '').lower(),
            'venue_capacity': show.get('venue_capacity'),
            'special_notes': show.get('special_notes', '')
        }
    
    def _convert_onsale_to_event_data(self, onsale: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ShowsOnSale onsale data to event analysis format"""
        return {
            'id': hash(f"{onsale.get('artist_name', '')}{onsale.get('venue_name', '')}{onsale.get('event_date', '')}"),
            'artist_name': onsale.get('artist_name', ''),
            'venue_name': onsale.get('venue_name', ''),
            'venue_city': onsale.get('venue_city', ''),
            'venue_state': onsale.get('venue_state', ''),
            'event_date': onsale.get('event_date', ''),
            'on_sale_date': onsale.get('on_sale_date', ''),
            'is_ticketmaster': onsale.get('ticket_source') == 'ticketmaster',
            'is_presale': onsale.get('is_presale', False),
            'presale_code_required': onsale.get('presale_code_required', False)
        }
    
    def _generate_summary_stats(self, recommendations: List[PurchaseRecommendation]) -> Dict[str, Any]:
        """Generate summary statistics for recommendations"""
        if not recommendations:
            return {}
        
        recommendation_counts = {}
        for rec in recommendations:
            rec_type = rec.recommendation.value
            recommendation_counts[rec_type] = recommendation_counts.get(rec_type, 0) + 1
        
        total_investment = sum(rec.max_recommended_investment for rec in recommendations)
        avg_confidence = sum(rec.confidence_score for rec in recommendations) / len(recommendations)
        avg_margin = sum(rec.expected_profit_margin for rec in recommendations) / len(recommendations)
        
        return {
            'total_events': len(recommendations),
            'recommendation_breakdown': recommendation_counts,
            'total_recommended_investment': total_investment,
            'average_confidence_score': avg_confidence,
            'average_expected_margin': avg_margin,
            'high_priority_count': recommendation_counts.get('strong_buy', 0),
            'medium_priority_count': recommendation_counts.get('selective_buy', 0)
        }
    
    def _generate_market_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate high-level market recommendations"""
        recommendations = []
        
        opportunities = report.get('upcoming_opportunities', [])
        if len(opportunities) > 5:
            recommendations.append("Strong market activity detected. Consider increasing inventory allocation.")
        elif len(opportunities) < 2:
            recommendations.append("Limited opportunities identified. Focus on market research for emerging trends.")
        
        hot_artists = report.get('hot_artists', [])
        if hot_artists:
            recommendations.append(f"Monitor tours for trending artists: {', '.join([artist['artist'] for artist in hot_artists[:3]])}")
        
        return recommendations


# Example usage and testing functions
def run_example_analysis():
    """Example function demonstrating system usage"""
    optimizer = TicketBrokerOptimizer()
    
    # Example event data
    example_event = {
        'artist_name': 'Taylor Swift',
        'venue_name': 'Madison Square Garden',
        'venue_city': 'New York',
        'venue_state': 'NY',
        'event_date': '2024-06-15T20:00:00Z',
        'on_sale_date': '2024-03-01T10:00:00Z',
        'face_value_min': 75.0,
        'face_value_max': 300.0,
        'tour_name': 'Eras Tour',
        'is_farewell_tour': False
    }
    
    # Perform analysis
    print("🎫 Starting Ticket Broker Analysis...")
    
    # Quick decision framework
    print("\n1. Quick Decision Framework (60-second check):")
    quick_decision = optimizer.quick_decision_framework(
        example_event['artist_name'],
        example_event['venue_name'],
        example_event['venue_city'],
        example_event['event_date']
    )
    print(f"   Decision: {quick_decision['decision']}")
    print(f"   Score: {quick_decision['quick_score']}")
    print(f"   Reasoning: {quick_decision['reasoning']}")
    
    # Full analysis
    print("\n2. Comprehensive Analysis:")
    recommendation = optimizer.analyze_event(example_event)
    print(f"   Recommendation: {recommendation.recommendation.value}")
    print(f"   Total Score: {recommendation.total_score}")
    print(f"   Confidence: {recommendation.confidence_score:.2%}")
    print(f"   Expected Margin: {recommendation.expected_profit_margin:.2%}")
    print(f"   Max Investment: ${recommendation.max_recommended_investment:,.2f}")
    print(f"   Key Factors: {', '.join(recommendation.key_factors)}")
    
    if recommendation.risk_factors:
        print(f"   Risk Factors: {', '.join(recommendation.risk_factors)}")
    
    print(f"\n   Reasoning: {recommendation.reasoning}")
    
    # Export report
    print("\n3. Exporting Analysis Report...")
    report_path = optimizer.export_analysis_report([recommendation])
    print(f"   Report saved to: {report_path}")
    
    print("\n✅ Analysis Complete!")


if __name__ == "__main__":
    # Run example analysis
    run_example_analysis()