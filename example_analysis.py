#!/usr/bin/env python3
"""
Ticket Broker Optimization System - Example Analysis

This script demonstrates how to use the ticket broker system
to analyze events and make investment decisions.

Run with: python example_analysis.py
"""

import sys
import os
from datetime import datetime, timedelta

# Add the ticket_broker package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ticket_broker import TicketBrokerOptimizer, RecommendationLevel


def main():
    """Run example analysis scenarios"""
    
    print("🎫 Ticket Broker Optimization System - Example Analysis")
    print("=" * 60)
    
    # Initialize the optimizer
    try:
        optimizer = TicketBrokerOptimizer()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed all dependencies: pip install -r requirements.txt")
        print("   2. Set up API keys in .env file")
        print("   3. Installed ChromeDriver")
        return
    
    # Define sample events for analysis
    sample_events = [
        {
            'artist_name': 'Taylor Swift',
            'venue_name': 'Madison Square Garden',
            'venue_city': 'New York',
            'venue_state': 'NY',
            'event_date': (datetime.now() + timedelta(days=90)).isoformat() + 'Z',
            'on_sale_date': (datetime.now() + timedelta(days=30)).isoformat() + 'Z',
            'face_value_min': 75.0,
            'face_value_max': 300.0,
            'tour_name': 'Eras Tour',
            'is_farewell_tour': False
        },
        {
            'artist_name': 'Bad Bunny',
            'venue_name': 'Crypto.com Arena',
            'venue_city': 'Los Angeles',
            'venue_state': 'CA',
            'event_date': (datetime.now() + timedelta(days=120)).isoformat() + 'Z',
            'on_sale_date': (datetime.now() + timedelta(days=45)).isoformat() + 'Z',
            'face_value_min': 89.0,
            'face_value_max': 250.0,
            'tour_name': 'Most Wanted Tour',
            'is_farewell_tour': False
        },
        {
            'artist_name': 'Billie Eilish',
            'venue_name': 'United Center',
            'venue_city': 'Chicago',
            'venue_state': 'IL',
            'event_date': (datetime.now() + timedelta(days=60)).isoformat() + 'Z',
            'on_sale_date': (datetime.now() + timedelta(days=15)).isoformat() + 'Z',
            'face_value_min': 55.0,
            'face_value_max': 180.0,
            'tour_name': 'Hit Me Hard and Soft Tour',
            'is_farewell_tour': False
        }
    ]
    
    print(f"\n📊 Analyzing {len(sample_events)} sample events...\n")
    
    # Example 1: Quick Decision Framework
    print("🚀 Example 1: Quick Decision Framework (60-second analysis)")
    print("-" * 50)
    
    for i, event in enumerate(sample_events[:2], 1):
        print(f"\n{i}. {event['artist_name']} at {event['venue_name']}")
        
        try:
            quick_decision = optimizer.quick_decision_framework(
                event['artist_name'],
                event['venue_name'],
                event['venue_city'],
                event['event_date']
            )
            
            print(f"   Decision: {quick_decision['decision'].upper()}")
            print(f"   Quick Score: {quick_decision['quick_score']}/6")
            print(f"   Reasoning: {quick_decision['reasoning']}")
            
            if quick_decision['next_steps']:
                print(f"   Next Steps: {', '.join(quick_decision['next_steps'])}")
                
        except Exception as e:
            print(f"   ❌ Error in quick analysis: {e}")
    
    # Example 2: Comprehensive Analysis
    print(f"\n\n🔍 Example 2: Comprehensive Event Analysis")
    print("-" * 50)
    
    try:
        # Analyze the first event in detail
        event = sample_events[0]
        print(f"\nAnalyzing: {event['artist_name']} at {event['venue_name']}")
        
        recommendation = optimizer.analyze_event(event)
        
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"   Recommendation: {recommendation.recommendation.value.upper()}")
        print(f"   Total Score: {recommendation.total_score} points")
        print(f"   Confidence Level: {recommendation.confidence_score:.1%}")
        print(f"   Expected Profit Margin: {recommendation.expected_profit_margin:.1%}")
        print(f"   Max Recommended Investment: ${recommendation.max_recommended_investment:,.2f}")
        
        if recommendation.key_factors:
            print(f"   Key Success Factors: {', '.join(recommendation.key_factors)}")
        
        if recommendation.risk_factors:
            print(f"   Risk Factors: {', '.join(recommendation.risk_factors)}")
        
        print(f"\n💭 Detailed Reasoning:")
        print(f"   {recommendation.reasoning}")
        
        if recommendation.action_required_by:
            print(f"\n⏰ Action Required By: {recommendation.action_required_by}")
        
        # Recommendation interpretation
        print(f"\n🎯 INVESTMENT DECISION:")
        if recommendation.recommendation == RecommendationLevel.STRONG_BUY:
            print("   ✅ STRONG BUY - High profit potential, proceed with confidence")
        elif recommendation.recommendation == RecommendationLevel.SELECTIVE_BUY:
            print("   ⚠️  SELECTIVE BUY - Moderate potential, invest carefully")
        elif recommendation.recommendation == RecommendationLevel.AVOID:
            print("   ❌ AVOID - Low profit potential, pass on this opportunity")
        else:
            print("   🚨 HIGH RISK - Significant risk factors, avoid investment")
            
    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")
    
    # Example 3: Batch Analysis
    print(f"\n\n📊 Example 3: Batch Analysis (Multiple Events)")
    print("-" * 50)
    
    try:
        print("Analyzing all sample events...")
        recommendations = optimizer.analyze_multiple_events(sample_events)
        
        print(f"\n🏆 RANKED OPPORTUNITIES:")
        for i, rec in enumerate(recommendations, 1):
            artist_name = next((e['artist_name'] for e in sample_events if e.get('id') == rec.event_id), 'Unknown')
            venue_name = next((e['venue_name'] for e in sample_events if e.get('id') == rec.event_id), 'Unknown')
            
            print(f"   {i}. {artist_name} at {venue_name}")
            print(f"      Recommendation: {rec.recommendation.value.upper()}")
            print(f"      Score: {rec.total_score} | Confidence: {rec.confidence_score:.1%} | Expected Margin: {rec.expected_profit_margin:.1%}")
            print()
        
        # Summary statistics
        total_investment = sum(rec.max_recommended_investment for rec in recommendations)
        strong_buys = len([r for r in recommendations if r.recommendation == RecommendationLevel.STRONG_BUY])
        selective_buys = len([r for r in recommendations if r.recommendation == RecommendationLevel.SELECTIVE_BUY])
        
        print(f"📈 PORTFOLIO SUMMARY:")
        print(f"   Strong Buy Opportunities: {strong_buys}")
        print(f"   Selective Buy Opportunities: {selective_buys}")
        print(f"   Total Recommended Investment: ${total_investment:,.2f}")
        
    except Exception as e:
        print(f"❌ Error in batch analysis: {e}")
    
    # Example 4: Export Report
    print(f"\n\n📄 Example 4: Export Analysis Report")
    print("-" * 50)
    
    try:
        if 'recommendations' in locals():
            report_path = optimizer.export_analysis_report(recommendations)
            print(f"✅ Analysis report exported to: {report_path}")
            print("   This JSON file contains detailed analysis data for further review.")
        else:
            print("❌ No recommendations available to export")
    except Exception as e:
        print(f"❌ Error exporting report: {e}")
    
    print(f"\n🎯 SYSTEM OVERVIEW:")
    print("   • This system analyzes events using multiple data sources")
    print("   • Scoring framework provides objective investment guidance")
    print("   • Risk assessment helps avoid losses")
    print("   • Consistent 40%+ profit margins are the goal")
    
    print(f"\n💡 NEXT STEPS:")
    print("   1. Set up your API keys in the .env file for full functionality")
    print("   2. Monitor upcoming on-sales with get_upcoming_onsales_analysis()")
    print("   3. Search specific artists with search_and_analyze_artist_tours()")
    print("   4. Generate market reports for your target cities")
    print("   5. Use the quick decision framework for rapid screening")
    
    print("\n✨ Happy ticket brokering! Remember: consistent profits beat home runs! 🎫💰")


if __name__ == "__main__":
    main()