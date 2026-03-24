"""Domain configs, site profiles, token budgets, and safety constants."""

SITE_PROFILES: dict[str, dict] = {
    "kitchensdirectory": {
        "domain": "kitchensdirectory.co.uk",
        "gsc_property": "https://www.kitchensdirectory.co.uk",
        "primary_topic": "UK kitchen makers and manufacturers directory",
        "target_audience": "homeowners looking for kitchen companies in the UK",
        "content_types": [
            "location_pages",
            "buyer_guides",
            "comparison_articles",
            "listicles",
        ],
        "seed_keywords": [
            "kitchen makers UK",
            "kitchen companies near me",
            "bespoke kitchen manufacturers",
            "handmade kitchen companies",
        ],
        "geo_focus": "UK",
        "monetisation": "advertising + leads to kitchen companies",
        "competitors": [
            "houzz.com/uk",
            "checkatrade.com",
            "trustatrader.com",
            "rated.co.uk",
            "thekitchenspecialist.co.uk",
        ],
    },
    "freeroomplanner": {
        "domain": "freeroomplanner.com",
        "gsc_property": "https://www.freeroomplanner.com",
        "primary_topic": "free online room planning tool",
        "target_audience": (
            "homeowners planning a kitchen, bathroom or living room renovation"
        ),
        "content_types": [
            "how_to_guides",
            "tool_landing_pages",
            "inspiration_articles",
        ],
        "seed_keywords": [
            "free room planner",
            "kitchen layout planner",
            "room planning tool online",
            "bathroom planner free",
        ],
        "geo_focus": "UK primary, global secondary",
        "monetisation": "lead capture → sell to kitchen/bathroom companies",
        "competitors": [
            "roomsketcher.com",
            "planner5d.com",
            "ikea.com/gb/en/room-planner",
            "magicplan.app",
        ],
    },
    "kitchen_estimator": {
        "domain": "TBD",
        "primary_topic": "kitchen price estimator and cost guide",
        "target_audience": "homeowners budgeting for a kitchen renovation",
        "content_types": [
            "cost_guides",
            "calculator_landing_pages",
            "comparison_articles",
        ],
        "seed_keywords": [
            "how much does a kitchen cost",
            "kitchen prices UK",
            "fitted kitchen cost",
            "kitchen installation cost",
        ],
        "geo_focus": "UK",
        "monetisation": "high-intent lead capture → premium pricing for leads",
        "competitors": [],
    },
}

# Token budgets — never exceed these per task call
TOKEN_BUDGETS: dict[str, int] = {
    "classify_prospect": 50,
    "extract_contact_email": 30,
    "score_prospect": 100,
    "summarise_page": 200,
    "generate_keyword_ideas": 500,
    "write_content_brief": 1500,
    "write_tier2_email": 300,
    "write_tier1_email": 400,
    "write_blog_post": 2500,
    "write_location_page": 1500,
    "write_followup_email": 250,
    "generate_pr_angles": 600,
    "weekly_report": 1000,
}

# Weekly LLM spend cap in USD — agent downgrades models at 80% of this
MAX_WEEKLY_SPEND_USD: float = 50.00

# ---------------------------------------------------------------------------
# Outreach safety limits — hardcoded, NOT configurable via env vars
# ---------------------------------------------------------------------------
MAX_DAILY_OUTREACH_EMAILS: int = 20
OUTREACH_SEND_WINDOW_START: int = 8  # 08:00 UK
OUTREACH_SEND_WINDOW_END: int = 17  # 17:00 UK
MIN_DAYS_BETWEEN_DOMAIN_CONTACTS: int = 90
MIN_OUTREACH_SCORE: int = 35
TIER1_OUTREACH_SCORE: int = 65

# Domains that must never receive commercial outreach
BLOCKED_TLD_SUFFIXES: list[str] = [".gov.uk", ".ac.uk"]

# Warm-up schedule: day-range → max emails/day
WARMUP_SCHEDULE: dict[str, int] = {
    "week_1": 5,
    "week_2": 10,
    "week_3_plus": 20,
}

# Bounce rate threshold — pause all sends if exceeded
MAX_BOUNCE_RATE_PERCENT: float = 3.0
