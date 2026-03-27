-- Migration 003: Outreach CRM tables
-- Run via Supabase Dashboard → SQL Editor
--
-- Stores kitchen/bathroom companies and interior designers as business
-- entities for email outreach. Designed to be agent-agnostic — any agent
-- (Ralf, future agents, or human operators) can read and write these tables.

-- ---------------------------------------------------------------------------
-- Core contacts table — companies and individuals to target
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS crm_contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    company_name TEXT NOT NULL,
    contact_name TEXT,
    contact_role TEXT,

    -- Contact details
    email TEXT,
    phone TEXT,
    website TEXT,

    -- Location
    city TEXT,
    region TEXT,
    postcode TEXT,
    country TEXT DEFAULT 'GB',

    -- Classification
    category TEXT NOT NULL,              -- kitchen_company, bathroom_company, interior_designer
    subcategory TEXT,                     -- showroom, manufacturer, fitter, supplier, freelance, studio, firm

    -- Social profiles
    instagram TEXT,
    facebook TEXT,
    linkedin TEXT,

    -- Outreach state
    outreach_status TEXT DEFAULT 'not_contacted',  -- not_contacted, contacted, replied, partnership_active, declined, blocked
    outreach_segment TEXT,               -- matches OUTREACH_SEGMENTS key in outreach_strategy.py

    -- Scoring
    score INTEGER DEFAULT 0,
    tier TEXT,                            -- tier_1, tier_2, tier_3

    -- Linkage to backlink prospects (optional)
    backlink_prospect_id UUID,

    -- Metadata
    source TEXT,                          -- manual, scraped, kitchen_makers_import, google_maps, etc.
    tags JSONB DEFAULT '[]',
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    last_contacted_at TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- Interaction history log — every touchpoint with a contact
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS crm_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL,            -- references crm_contacts(id)

    -- Interaction details
    interaction_type TEXT NOT NULL,       -- email_sent, email_received, phone_call, meeting, note, social_dm
    direction TEXT DEFAULT 'outbound',   -- outbound, inbound, internal
    channel TEXT DEFAULT 'email',
    subject TEXT,
    body_preview TEXT,

    -- State
    status TEXT DEFAULT 'logged',        -- logged, pending_reply, replied, no_response

    -- Who performed this action (supports multi-agent use)
    performed_by TEXT DEFAULT 'ralf',

    created_at TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Kitchen makers directory (fixes missing schema for get_makers_by_location)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kitchen_makers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    city TEXT,
    region TEXT,
    postcode TEXT,
    country TEXT DEFAULT 'GB',
    founded INTEGER,
    website TEXT,
    email TEXT,
    phone TEXT,
    description TEXT,
    style_categories JSONB DEFAULT '[]',
    budget_tier TEXT,
    verified BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_crm_contacts_category ON crm_contacts(category);
CREATE INDEX IF NOT EXISTS idx_crm_contacts_city ON crm_contacts(city);
CREATE INDEX IF NOT EXISTS idx_crm_contacts_status ON crm_contacts(outreach_status);
CREATE INDEX IF NOT EXISTS idx_crm_contacts_email ON crm_contacts(email);
CREATE INDEX IF NOT EXISTS idx_crm_contacts_segment ON crm_contacts(outreach_segment);
CREATE INDEX IF NOT EXISTS idx_crm_interactions_contact ON crm_interactions(contact_id);
CREATE INDEX IF NOT EXISTS idx_crm_interactions_type ON crm_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_kitchen_makers_city ON kitchen_makers(city);
