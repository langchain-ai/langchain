"""Seed the CRM with sample contacts for development and demo purposes.

Usage::

    python -m agents.seo_agent.seed_crm
"""

from __future__ import annotations

import logging
import sys

from agents.seo_agent.tools.crm_tools import add_crm_contact, get_crm_contacts

logger = logging.getLogger(__name__)

SEED_CONTACTS: list[dict] = [
    # -- Kitchen companies --
    {
        "company_name": "Roundhouse Design",
        "category": "kitchen_company",
        "subcategory": "bespoke_maker",
        "website": "https://roundhousedesign.com",
        "email": "info@roundhousedesign.com",
        "city": "London",
        "region": "Greater London",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 85,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Harvey Jones Kitchens",
        "category": "kitchen_company",
        "subcategory": "showroom",
        "website": "https://harveyjones.com",
        "email": "hello@harveyjones.com",
        "city": "London",
        "region": "Greater London",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 78,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Tom Howley",
        "category": "kitchen_company",
        "subcategory": "bespoke_maker",
        "website": "https://tomhowley.co.uk",
        "email": "enquiries@tomhowley.co.uk",
        "city": "Congleton",
        "region": "Cheshire",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 82,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Sigma 3 Kitchens",
        "category": "kitchen_company",
        "subcategory": "manufacturer",
        "website": "https://sigma3.co.uk",
        "email": "info@sigma3.co.uk",
        "city": "Cardiff",
        "region": "South Wales",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 65,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Masterclass Kitchens",
        "category": "kitchen_company",
        "subcategory": "manufacturer",
        "website": "https://masterclasskitchens.co.uk",
        "email": "info@masterclasskitchens.co.uk",
        "city": "Llantrisant",
        "region": "South Wales",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 70,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Wren Kitchens",
        "category": "kitchen_company",
        "subcategory": "showroom",
        "website": "https://wrenkitchens.com",
        "email": "info@wrenkitchens.com",
        "city": "Barton-upon-Humber",
        "region": "Lincolnshire",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 55,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Naked Kitchens",
        "category": "kitchen_company",
        "subcategory": "bespoke_maker",
        "website": "https://nakedkitchens.com",
        "email": "hello@nakedkitchens.com",
        "city": "Norwich",
        "region": "Norfolk",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 72,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Second Nature Kitchens",
        "category": "kitchen_company",
        "subcategory": "manufacturer",
        "website": "https://sncollection.co.uk",
        "email": "info@sncollection.co.uk",
        "city": "Nelson",
        "region": "Lancashire",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 60,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Davonport Kitchens",
        "category": "kitchen_company",
        "subcategory": "bespoke_maker",
        "website": "https://davonport.com",
        "email": "info@davonport.com",
        "city": "Colchester",
        "region": "Essex",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 68,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Olive & Barr",
        "category": "kitchen_company",
        "subcategory": "bespoke_maker",
        "website": "https://oliveandbarr.com",
        "email": "hello@oliveandbarr.com",
        "city": "Ledbury",
        "region": "Herefordshire",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 74,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Benchmarx Kitchens",
        "category": "kitchen_company",
        "subcategory": "supplier",
        "website": "https://benchmarxkitchens.co.uk",
        "email": "info@benchmarxkitchens.co.uk",
        "city": "Manchester",
        "region": "Greater Manchester",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 45,
        "tier": "tier_3",
        "source": "seed",
    },
    {
        "company_name": "Handmade Kitchens Edinburgh",
        "category": "kitchen_company",
        "subcategory": "fitter",
        "website": "https://handmadekitchensedinburgh.co.uk",
        "email": "info@handmadekitchensedinburgh.co.uk",
        "city": "Edinburgh",
        "region": "Scotland",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 50,
        "tier": "tier_3",
        "source": "seed",
    },
    # -- Bathroom companies --
    {
        "company_name": "CP Hart Bathrooms",
        "category": "bathroom_company",
        "subcategory": "showroom",
        "website": "https://cphart.co.uk",
        "email": "info@cphart.co.uk",
        "city": "London",
        "region": "Greater London",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 80,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Lusso Stone",
        "category": "bathroom_company",
        "subcategory": "supplier",
        "website": "https://lussostone.com",
        "email": "info@lussostone.com",
        "city": "Manchester",
        "region": "Greater Manchester",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 66,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Ripples Bathrooms",
        "category": "bathroom_company",
        "subcategory": "showroom",
        "website": "https://qurantadue.com",
        "email": "info@qurantadue.com",
        "city": "Bath",
        "region": "Somerset",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 62,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Boundary Bathrooms",
        "category": "bathroom_company",
        "subcategory": "supplier",
        "website": "https://boundarybathrooms.co.uk",
        "email": "hello@boundarybathrooms.co.uk",
        "city": "Bristol",
        "region": "Avon",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 48,
        "tier": "tier_3",
        "source": "seed",
    },
    {
        "company_name": "Sanctuary Bathrooms",
        "category": "bathroom_company",
        "subcategory": "designer",
        "website": "https://sanctuarybathrooms.co.uk",
        "email": "info@sanctuarybathrooms.co.uk",
        "city": "Leeds",
        "region": "West Yorkshire",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 58,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Drench Bathrooms",
        "category": "bathroom_company",
        "subcategory": "supplier",
        "website": "https://drench.co.uk",
        "email": "hello@drench.co.uk",
        "city": "Birmingham",
        "region": "West Midlands",
        "country": "GB",
        "outreach_segment": "kitchen_bathroom_providers",
        "score": 52,
        "tier": "tier_3",
        "source": "seed",
    },
    # -- Interior designers --
    {
        "company_name": "Studio Duggan",
        "category": "interior_designer",
        "subcategory": "studio",
        "website": "https://studioduggan.com",
        "email": "hello@studioduggan.com",
        "city": "London",
        "region": "Greater London",
        "country": "GB",
        "outreach_segment": "home_interior_bloggers",
        "score": 76,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Sims Hilditch",
        "category": "interior_designer",
        "subcategory": "firm",
        "website": "https://simshilditch.com",
        "email": "info@simshilditch.com",
        "city": "Marlborough",
        "region": "Wiltshire",
        "country": "GB",
        "outreach_segment": "home_interior_bloggers",
        "score": 71,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Anna Sheridan Interiors",
        "category": "interior_designer",
        "subcategory": "freelance",
        "website": "https://annasheridaninteriors.co.uk",
        "email": "anna@annasheridaninteriors.co.uk",
        "city": "Glasgow",
        "region": "Scotland",
        "country": "GB",
        "outreach_segment": "home_interior_bloggers",
        "score": 55,
        "tier": "tier_2",
        "source": "seed",
    },
    {
        "company_name": "Barlow & Barlow",
        "category": "interior_designer",
        "subcategory": "firm",
        "website": "https://barlowandbarlow.com",
        "email": "info@barlowandbarlow.com",
        "city": "London",
        "region": "Greater London",
        "country": "GB",
        "outreach_segment": "home_interior_bloggers",
        "score": 64,
        "tier": "tier_2",
        "source": "seed",
    },
    # -- Bloggers --
    {
        "company_name": "Mad About The House",
        "category": "blogger",
        "subcategory": "home_improvement",
        "contact_name": "Kate Watson-Smyth",
        "website": "https://madaboutthehouse.com",
        "email": "kate@madaboutthehouse.com",
        "city": "London",
        "region": "Greater London",
        "country": "GB",
        "outreach_segment": "home_interior_bloggers",
        "score": 88,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "Dear Designer",
        "category": "blogger",
        "subcategory": "lifestyle",
        "contact_name": "Carole King",
        "website": "https://deardesigner.co.uk",
        "email": "carole@deardesigner.co.uk",
        "city": "London",
        "region": "Greater London",
        "country": "GB",
        "outreach_segment": "home_interior_bloggers",
        "score": 72,
        "tier": "tier_1",
        "source": "seed",
    },
    {
        "company_name": "The Frugality",
        "category": "blogger",
        "subcategory": "lifestyle",
        "contact_name": "Alex Sheridan",
        "website": "https://thefrugality.com",
        "email": "alex@thefrugality.com",
        "city": "Manchester",
        "region": "Greater Manchester",
        "country": "GB",
        "outreach_segment": "home_interior_bloggers",
        "score": 60,
        "tier": "tier_2",
        "source": "seed",
    },
]


def seed_crm() -> int:
    """Insert sample contacts into the CRM, skipping any that already exist.

    Returns:
        Number of new contacts inserted.
    """
    existing = get_crm_contacts(limit=500)
    existing_names = {c.get("company_name", "").lower() for c in existing}

    inserted = 0
    for contact in SEED_CONTACTS:
        if contact["company_name"].lower() in existing_names:
            logger.info("Skipping %s (already exists)", contact["company_name"])
            continue

        try:
            add_crm_contact(**contact)
            inserted += 1
            logger.info("Added %s", contact["company_name"])
        except Exception:
            logger.error(
                "Failed to add %s", contact["company_name"], exc_info=True
            )

    return inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = seed_crm()
    print(f"\nSeeded {count} new contacts ({len(SEED_CONTACTS) - count} skipped).")
    sys.exit(0)
