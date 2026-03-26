"""Domain configs, site profiles, token budgets, and safety constants."""

SITE_PROFILES: dict[str, dict] = {
    "kitchensdirectory": {
        "domain": "kitchensdirectory.co.uk",
        "gsc_property": "https://www.kitchensdirectory.co.uk",
        "status": "upcoming",  # Not yet actively doing SEO — focus on freeroomplanner first
        "primary_topic": "UK kitchen makers and manufacturers directory",
        "description": (
            "The independent directory of Britain's handmade kitchen makers. "
            "159+ verified makers listed, individually researched. Covers 11 style "
            "categories across 4 budget tiers (under £10k to £55k+). Also includes "
            "worktop suppliers directory, cooker brands (AGA, Rangemaster, Rayburn, Falcon), "
            "kitchen inspiration gallery, and buyer FAQs. Makers don't self-register — "
            "every listing meets the same editorial standard."
        ),
        "target_audience": (
            "UK homeowners looking for handmade/bespoke kitchen makers, "
            "from budget (under £10k) to heritage/ultra-luxury (£55k+)"
        ),
        "value_proposition": (
            "Connects buyers with verified independent makers who lack marketing budgets. "
            "Handmade kitchens can cost less than equivalent mass-produced ones. "
            "Transparent pricing by scope of supply."
        ),
        "content_types": [
            "location_pages",
            "buyer_guides",
            "comparison_articles",
            "listicles",
            "kitchen_inspiration",
            "style_guides",
        ],
        "seed_keywords": [
            "kitchen makers UK",
            "kitchen companies near me",
            "bespoke kitchen manufacturers",
            "handmade kitchen companies",
            "shaker kitchen makers",
            "kitchen makers Manchester",
            "kitchen makers London",
        ],
        "geo_focus": "UK",
        "monetisation": "advertising + leads to kitchen companies",
        "competitors": [
            "houzz.com",
            "checkatrade.com",
            "trustatrader.com",
            "rated.co.uk",
            "thekitchenspecialist.co.uk",
        ],
        "cross_links": {
            "freeroomplanner": "Design Your Kitchen → free room planner CTA",
            "kitchen_estimator": "How Much Does a Kitchen Cost → cost estimator CTA",
        },
    },
    "freeroomplanner": {
        "domain": "freeroomplanner.com",
        "gsc_property": "https://www.freeroomplanner.com",
        "status": "active",  # Primary focus site
        "primary_topic": "free online room planning tool",
        "description": (
            "Free browser-based floor planner for homeowners. Draw walls with snap-to-grid "
            "(10cm), add 30+ furniture items (drag and resize), live measurements in metres/cm/feet, "
            "export as PNG. No sign-up, no email, no download required. Room-specific tools: "
            "kitchen planner, bathroom planner, bedroom planner, living room planner, "
            "and multi-room floor plan maker. Built with React and Canvas."
        ),
        "target_audience": (
            "Homeowners planning a kitchen, bathroom, bedroom, or living room renovation. "
            "Also used by kitchen makers, bathroom fitters, architects, and contractors "
            "who receive plans from their clients."
        ),
        "value_proposition": (
            "Completely free — no trial, no credit card, no premium tier. "
            "Accurate enough to share with professionals (10cm grid snap, real-time measurements). "
            "No install needed — runs in browser. You arrive with a plan. They arrive prepared."
        ),
        "content_types": [
            "how_to_guides",
            "tool_landing_pages",
            "room_planning_tutorials",
            "inspiration_articles",
            "comparison_articles",
        ],
        "seed_keywords": [
            "free room planner",
            "kitchen layout planner",
            "room planning tool online",
            "bathroom planner free",
            "floor plan maker",
            "room design tool",
            "kitchen planner free",
        ],
        "geo_focus": "UK primary, global secondary",
        "monetisation": "lead capture → sell to kitchen/bathroom companies",
        "competitors": [
            "roomsketcher.com",
            "planner5d.com",
            "ikea.com",
            "magicplan.app",
            "floorplanner.com",
        ],
        "cross_links": {
            "kitchensdirectory": "Find a kitchen maker → directory CTA",
            "kitchen_estimator": "How much will it cost? → estimator CTA",
        },
    },
    "ralf_seo": {
        "domain": "ralf-seo.vercel.app",  # placeholder until custom domain
        "status": "active",
        "description": "Ralf's personal SEO field journal",
        "seed_keywords": [],  # Not doing keyword-targeted SEO for this site
        "competitors": [],
    },
    "kitchen_estimator": {
        "domain": "kitchencostestimator.com",
        "gsc_property": "https://www.kitchencostestimator.com",
        "status": "upcoming",  # Not yet actively doing SEO — focus on freeroomplanner first
        "primary_topic": "kitchen price estimator and cost guide",
        "description": (
            "Interactive kitchen renovation cost estimator for UK, US, and Canada. "
            "Step-by-step wizard: country selection (auto-detected by IP), kitchen size and layout "
            "(with SVG floor plan icons), scope of works checklist, units & finish tier, "
            "worktop material, per-appliance budget selection, flooring, trade work & extras. "
            "Shows Low/Mid/High cost range with stacked bar chart and itemised breakdown. "
            "68 cost items and 26 multipliers from real pricing data. "
            "Built with Next.js 16, Supabase backend, deployed on Vercel."
        ),
        "target_audience": "homeowners budgeting for a kitchen renovation in UK, US, or Canada",
        "value_proposition": (
            "Accurate, data-driven kitchen cost estimates in 2 minutes. "
            "Country-specific pricing, real-time calculation, no sign-up required."
        ),
        "content_types": [
            "cost_guides",
            "calculator_landing_pages",
            "comparison_articles",
            "regional_cost_pages",
        ],
        "seed_keywords": [
            "how much does a kitchen cost",
            "kitchen prices UK",
            "fitted kitchen cost",
            "kitchen installation cost",
            "kitchen renovation cost calculator",
            "average kitchen cost UK 2026",
        ],
        "geo_focus": "UK primary, US and Canada secondary",
        "monetisation": "high-intent lead capture → premium pricing for leads",
        "competitors": [
            "checkatrade.com",
            "householdquotes.co.uk",
            "mybuilder.com",
        ],
        "cross_links": {
            "freeroomplanner": "Plan your layout → free room planner CTA",
            "kitchensdirectory": "Find a kitchen maker → directory CTA",
        },
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
