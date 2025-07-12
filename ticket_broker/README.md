# Ticket Broker Optimization System

A comprehensive Python system for analyzing live event tickets on the secondary market to optimize buying decisions with 40%+ profit margins.

## 🎯 Overview

This system implements a systematic approach to evaluate event profitability before purchasing tickets, using multiple data sources and a comprehensive scoring framework. It helps solo ticket brokers make informed decisions about which events to invest in for maximum profitability.

## 🚀 Features

### Data Collection & Analysis
- **Billboard Charts Integration**: Real-time chart positions, momentum tracking, and industry trends
- **Spotify API**: Artist popularity, streaming numbers, album releases, and trend analysis
- **ShowsOnSale.com Integration**: Tour announcements, on-sale dates, venue information, and market trends
- **Economic Data**: Local market conditions, unemployment rates, and consumer spending indicators
- **Social Media Analysis**: Buzz tracking, sentiment analysis, and fan engagement metrics

### Comprehensive Scoring Framework
- **High Profit Indicators** (+3 points each): First tour in 2+ years, venue under 5K capacity, sold out under 1 hour, major awards, farewell tours
- **Medium Profit Indicators** (+2 points each): New album within 6 months, premium venues, strong local fanbase, weekend shows, holiday dates
- **Low Profit Indicators** (+1 point each): Established acts, mid-tier venues, moderate buzz, standard prices
- **Risk Factors** (-2 points each): Frequent touring, large venues (20K+), competing events, declining popularity

### Intelligent Decision Making
- **60-Second Gut Check**: Quick preliminary analysis for rapid decision making
- **15-minute Deep Dive**: Comprehensive analysis with detailed reasoning
- **Market Analysis**: Local conditions, competition, and economic factors
- **Financial Projections**: Expected profit margins and investment recommendations

## 📊 Scoring Guide

- **15+ points**: High profit potential - **STRONG BUY**
- **10-14 points**: Moderate profit potential - **SELECTIVE BUY**
- **5-9 points**: Low profit potential - **AVOID**
- **Under 5 points**: High risk - **AVOID**

## 🛠 Installation

### Prerequisites
- Python 3.9+
- Chrome/Chromium browser (for web scraping)
- ChromeDriver (for Selenium automation)

### 1. Install Dependencies
```bash
# Clone or download the project
cd ticket_broker

# Install required packages
pip install -r requirements.txt
```

### 2. Set Up API Keys
Copy the example environment file and add your API credentials:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

#### Required APIs:
- **Spotify**: [Get API keys](https://developer.spotify.com/)
- **Twitter** (optional): [Get API keys](https://developer.twitter.com/)
- **Ticketmaster** (optional): [Get API key](https://developer.ticketmaster.com/)

#### Optional APIs for Enhanced Analysis:
- **FRED Economic Data**: [Get API key](https://fred.stlouisfed.org/docs/api/)
- **Alpha Vantage**: [Get API key](https://www.alphavantage.co/)

### 3. Install ChromeDriver
```bash
# On macOS with Homebrew
brew install chromedriver

# On Ubuntu/Debian
sudo apt-get install chromium-chromedriver

# On Windows - download from https://chromedriver.chromium.org/
```

## 📖 Usage

### Quick Start Example

```python
from ticket_broker import TicketBrokerOptimizer

# Initialize the system
optimizer = TicketBrokerOptimizer()

# Define an event to analyze
event_data = {
    'artist_name': 'Taylor Swift',
    'venue_name': 'Madison Square Garden',
    'venue_city': 'New York',
    'venue_state': 'NY',
    'event_date': '2024-06-15T20:00:00Z',
    'on_sale_date': '2024-03-01T10:00:00Z',
    'face_value_min': 75.0,
    'face_value_max': 300.0,
    'tour_name': 'Eras Tour'
}

# Perform quick decision check (60 seconds)
quick_decision = optimizer.quick_decision_framework(
    event_data['artist_name'],
    event_data['venue_name'],
    event_data['venue_city'],
    event_data['event_date']
)

print(f"Quick Decision: {quick_decision['decision']}")
print(f"Reasoning: {quick_decision['reasoning']}")

# Perform full analysis (if quick check is positive)
if quick_decision['decision'] in ['proceed_with_analysis', 'investigate_further']:
    recommendation = optimizer.analyze_event(event_data)
    
    print(f"\nRecommendation: {recommendation.recommendation.value}")
    print(f"Total Score: {recommendation.total_score}")
    print(f"Confidence: {recommendation.confidence_score:.1%}")
    print(f"Expected Profit Margin: {recommendation.expected_profit_margin:.1%}")
    print(f"Max Investment: ${recommendation.max_recommended_investment:,.2f}")
    print(f"Reasoning: {recommendation.reasoning}")
```

### Advanced Usage

#### 1. Analyze Multiple Events
```python
events = [event1_data, event2_data, event3_data]
recommendations = optimizer.analyze_multiple_events(events)

# Results are automatically sorted by priority
for rec in recommendations:
    print(f"{rec.recommendation.value}: Score {rec.total_score}")
```

#### 2. Search and Analyze Artist Tours
```python
# Find and analyze all upcoming shows for an artist
recommendations = optimizer.search_and_analyze_artist_tours("Billie Eilish")

for rec in recommendations:
    if rec.recommendation.value in ['strong_buy', 'selective_buy']:
        print(f"Opportunity found: {rec.reasoning}")
```

#### 3. Monitor Upcoming On-Sales
```python
# Get upcoming on-sales for the next 7 days
priority_onsales = optimizer.get_upcoming_onsales_analysis(days_ahead=7)

print(f"Found {len(priority_onsales)} priority opportunities")
for opportunity in priority_onsales:
    print(f"- {opportunity.reasoning}")
```

#### 4. Generate Market Reports
```python
# Generate market report for a specific city
market_report = optimizer.generate_market_report(city="Los Angeles")

print("Hot Artists:", market_report['hot_artists'])
print("Upcoming Opportunities:", len(market_report['upcoming_opportunities']))
print("Market Recommendations:", market_report['recommendations'])
```

#### 5. Export Analysis Reports
```python
# Export detailed analysis to JSON
report_path = optimizer.export_analysis_report(recommendations)
print(f"Report saved to: {report_path}")
```

## 🔧 Configuration

### Business Rules (configurable in .env):
- `MINIMUM_PROFIT_MARGIN`: Minimum acceptable profit margin (default: 40%)
- `MAX_INVESTMENT_PER_EVENT`: Maximum investment per single event
- `MAX_TOTAL_INVENTORY_VALUE`: Maximum total portfolio value

### Scoring Thresholds:
- `HIGH_PROFIT_SCORE_THRESHOLD`: Score for strong buy recommendation (default: 15)
- `MODERATE_PROFIT_SCORE_THRESHOLD`: Score for selective buy (default: 10)
- `LOW_PROFIT_SCORE_THRESHOLD`: Score for avoid recommendation (default: 5)

## 🎯 Decision Framework

### 60-Second Gut Check
1. **Artist Momentum**: Is the artist having a moment?
2. **Market Tier**: Is this a strong market?
3. **Recent Activity**: New releases or awards?
4. **Quick Score**: Calculate preliminary score

### 5-Minute Research
If gut check is positive:
1. Check venue capacity and type
2. Verify tour frequency
3. Analyze local market strength
4. Review competition

### 15-Minute Deep Dive
For promising opportunities:
1. Comprehensive scoring analysis
2. Financial projections
3. Risk assessment
4. Investment recommendations

## 📊 Market Research Components

### 1. Artist/Team Performance Analysis
- ✅ Popularity trends (Spotify, Billboard)
- ✅ Tour frequency analysis
- ✅ Last local performance date
- ✅ Fan demographics and loyalty
- ✅ Career trajectory assessment

### 2. Venue & Market Factors
- ✅ Venue size and prestige analysis
- ✅ Local market strength evaluation
- ✅ Competition assessment
- ✅ Accessibility and parking factors

### 3. Timing & Demand Indicators
- ✅ Sellout speed tracking
- ✅ Social media buzz analysis
- ✅ Seasonal demand patterns
- ✅ Special circumstances (farewell tours, anniversaries)

### 4. Pricing Intelligence
- ✅ Face value analysis
- ✅ Historical resale comparisons
- ✅ Market elasticity assessment
- ✅ Platform fee considerations

## 🚨 Red Flags to Avoid

The system automatically identifies and flags:
- Artists who tour constantly (over-saturated market)
- Events with very high face values relative to market
- Venues with poor reputations or difficult access
- Markets with weak economic conditions
- Events with significant competition same weekend
- Tours with mixed reviews or controversy

## 📈 Success Metrics

Track your success with built-in analytics:
- Recommendation accuracy rates
- Profit margin achievements
- Portfolio performance monitoring
- Market trend identification
- Risk factor effectiveness

## 🔮 Future Enhancements

Planned features include:
- Real-time inventory tracking
- Automated purchase execution
- Machine learning model improvements
- Mobile app interface
- Advanced portfolio optimization
- Integration with ticket marketplace APIs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This system is for educational and research purposes. Users are responsible for:
- Complying with all applicable laws and regulations
- Respecting terms of service for all integrated APIs
- Making their own investment decisions
- Understanding the risks involved in ticket speculation

The system provides analysis and recommendations but does not guarantee profits. Always conduct your own due diligence before making investment decisions.

## 🆘 Support

For questions, issues, or feature requests, please open an issue on GitHub or contact the development team.

---

**Remember**: The goal is consistent profitability, not home runs. Small, consistent profits beat big losses every time! 🎫💰