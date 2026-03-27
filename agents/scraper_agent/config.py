"""Scraper agent configuration."""

import os

# Firecrawl
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "").strip()

# Supabase (shared with SEO agent)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("SCRAPER_TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_OWNER_CHAT_ID = os.environ.get("TELEGRAM_OWNER_CHAT_ID", "").strip()

# Scheduling
DAILY_TARGET = int(os.environ.get("SCRAPER_DAILY_TARGET", "50"))
SCRAPE_INTERVAL_HOURS = int(os.environ.get("SCRAPE_INTERVAL_HOURS", "6"))  # Run every 6 hours, ~12-13 per batch

# Search queries by country and category
SEARCH_QUERIES = {
    "UK": {
        "kitchen_company": [
            "kitchen showroom UK",
            "fitted kitchen company UK",
            "bespoke kitchen maker UK",
            "kitchen installation company UK",
            "kitchen design studio UK",
            "kitchen company London",
            "kitchen showroom Manchester",
            "kitchen fitter Birmingham",
            "kitchen company Leeds",
            "kitchen maker Bristol",
            "kitchen showroom Glasgow",
            "kitchen company Edinburgh",
            "fitted kitchen Liverpool",
            "kitchen designer Newcastle",
            "kitchen company Sheffield",
            "kitchen showroom Nottingham",
            "kitchen fitter Cardiff",
            "kitchen company Brighton",
            "kitchen showroom Southampton",
            "kitchen maker York",
            "kitchen company Oxford",
            "kitchen showroom Cambridge",
            "kitchen fitter Leicester",
            "kitchen company Plymouth",
            "kitchen showroom Belfast",
        ],
        "bathroom_company": [
            "bathroom showroom UK",
            "bathroom fitting company UK",
            "bathroom renovation company UK",
            "bathroom design specialist UK",
            "bathroom installer UK",
            "bathroom company London",
            "bathroom showroom Manchester",
            "bathroom fitter Birmingham",
            "bathroom company Leeds",
            "bathroom showroom Bristol",
            "bathroom company Glasgow",
            "bathroom fitter Edinburgh",
        ],
    },
    "US": {
        "kitchen_company": [
            "kitchen remodeling company",
            "kitchen cabinet company",
            "kitchen design firm",
            "kitchen renovation contractor",
            "custom kitchen company",
            "kitchen remodeling New York",
            "kitchen company Los Angeles",
            "kitchen renovation Chicago",
            "kitchen remodeling Houston",
            "kitchen company Phoenix",
            "kitchen design San Francisco",
            "kitchen renovation Miami",
            "kitchen company Dallas",
            "kitchen remodeling Seattle",
            "kitchen company Denver",
            "kitchen renovation Atlanta",
            "kitchen company Boston",
            "kitchen remodeling Philadelphia",
            "kitchen company San Diego",
            "kitchen renovation Portland",
        ],
        "bathroom_company": [
            "bathroom remodeling company",
            "bathroom renovation contractor",
            "bathroom design company",
            "bathroom remodeling New York",
            "bathroom company Los Angeles",
            "bathroom renovation Chicago",
            "bathroom company Houston",
            "bathroom remodeling Miami",
            "bathroom company Dallas",
            "bathroom renovation Seattle",
        ],
    },
    "CA": {
        "kitchen_company": [
            "kitchen renovation company Canada",
            "kitchen cabinet maker Toronto",
            "kitchen company Vancouver",
            "kitchen renovation Calgary",
            "kitchen fitter Montreal",
            "kitchen company Ottawa",
            "kitchen renovation Edmonton",
            "kitchen company Winnipeg",
            "kitchen maker Halifax",
            "kitchen company Victoria",
        ],
        "bathroom_company": [
            "bathroom renovation company Canada",
            "bathroom company Toronto",
            "bathroom renovation Vancouver",
            "bathroom company Calgary",
            "bathroom fitter Montreal",
            "bathroom company Ottawa",
        ],
    },
}

SKIP_DOMAINS = {
    # Major retailers
    "bq.co.uk", "diy.com", "ikea.com", "wickes.co.uk", "howdens.com",
    "magnet.co.uk", "wren.co.uk", "homedepot.com", "lowes.com",
    "menards.com", "rona.ca",
    # Directories and marketplaces
    "checkatrade.com", "mybuilder.com", "trustatrader.com", "bark.com",
    "yell.com", "yelp.com", "houzz.com", "houzz.co.uk", "thumbtack.com",
    "homeadvisor.com", "angi.com", "angieslist.com", "homestars.com",
    "yellowpages.com", "yellowpages.ca", "192.com", "thomsonlocal.com",
    # Social / generic
    "facebook.com", "instagram.com", "pinterest.com", "twitter.com", "x.com",
    "linkedin.com", "youtube.com", "google.com", "amazon.com",
    "amazon.co.uk", "ebay.com", "ebay.co.uk", "tiktok.com",
    # Wikipedia / reference
    "wikipedia.org", "wikihow.com",
    # Our own sites
    "freeroomplanner.com", "kitchencostestimator.com", "kitchensdirectory.co.uk",
    "ralfseo.com",
}
