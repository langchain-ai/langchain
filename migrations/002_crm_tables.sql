-- Migration 002: CRM & tracking tables
-- Run via Supabase Dashboard → SQL Editor

-- Competitor ranking snapshots over time
CREATE TABLE IF NOT EXISTS seo_competitor_rankings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_domain TEXT NOT NULL,
    keyword TEXT NOT NULL,
    position INTEGER,
    volume INTEGER,
    traffic INTEGER,
    target_site TEXT NOT NULL,
    snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Content performance tracking (which posts drive traffic)
CREATE TABLE IF NOT EXISTS seo_content_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    title TEXT,
    target_site TEXT NOT NULL,
    target_keyword TEXT,
    organic_traffic INTEGER DEFAULT 0,
    keywords_ranking INTEGER DEFAULT 0,
    backlinks INTEGER DEFAULT 0,
    best_position INTEGER,
    snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Prospect communication log (CRM for outreach contacts)
CREATE TABLE IF NOT EXISTS seo_prospect_communications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prospect_id UUID REFERENCES seo_backlink_prospects(id),
    domain TEXT NOT NULL,
    contact_name TEXT,
    contact_email TEXT,
    contact_role TEXT,
    channel TEXT DEFAULT 'email',
    direction TEXT NOT NULL DEFAULT 'outbound',
    subject TEXT,
    body_preview TEXT,
    status TEXT DEFAULT 'sent',
    sent_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    replied_at TIMESTAMPTZ,
    follow_up_scheduled_at TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Keyword cache — local store so we don't re-query Ahrefs for known keywords
CREATE TABLE IF NOT EXISTS seo_keyword_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyword TEXT NOT NULL,
    country TEXT DEFAULT 'gb',
    volume INTEGER,
    difficulty INTEGER,
    cpc REAL,
    traffic_potential INTEGER,
    intent TEXT,
    last_updated TIMESTAMPTZ DEFAULT now(),
    source TEXT DEFAULT 'ahrefs',
    UNIQUE(keyword, country)
);

-- Our ranking positions over time (for our own sites)
CREATE TABLE IF NOT EXISTS seo_our_rankings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_site TEXT NOT NULL,
    keyword TEXT NOT NULL,
    position INTEGER,
    url TEXT,
    previous_position INTEGER,
    change INTEGER,
    volume INTEGER,
    snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_competitor_rankings_keyword ON seo_competitor_rankings(keyword, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_competitor_rankings_domain ON seo_competitor_rankings(competitor_domain);
CREATE INDEX IF NOT EXISTS idx_competitor_rankings_site ON seo_competitor_rankings(target_site);
CREATE INDEX IF NOT EXISTS idx_content_perf_url ON seo_content_performance(url);
CREATE INDEX IF NOT EXISTS idx_content_perf_site ON seo_content_performance(target_site, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_prospect_comms_prospect ON seo_prospect_communications(prospect_id);
CREATE INDEX IF NOT EXISTS idx_prospect_comms_domain ON seo_prospect_communications(domain);
CREATE INDEX IF NOT EXISTS idx_prospect_comms_status ON seo_prospect_communications(status);
CREATE INDEX IF NOT EXISTS idx_keyword_cache_keyword ON seo_keyword_cache(keyword, country);
CREATE INDEX IF NOT EXISTS idx_our_rankings_keyword ON seo_our_rankings(keyword, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_our_rankings_site ON seo_our_rankings(target_site, snapshot_date DESC);
