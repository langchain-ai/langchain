-- Migration 001: Create all SEO agent tables
-- Run via: psql $DATABASE_URL -f migrations/001_create_tables.sql
-- Or paste into Supabase Dashboard → SQL Editor

-- Helper function for running dynamic SQL (used by supabase_tools.py ensure_tables)
CREATE OR REPLACE FUNCTION exec_sql(query text) RETURNS void AS $$
BEGIN EXECUTE query; END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- SEO Keyword Opportunities
CREATE TABLE IF NOT EXISTS seo_keyword_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyword TEXT NOT NULL,
    volume INTEGER,
    kd INTEGER,
    cpc REAL,
    intent TEXT,
    current_position REAL,
    target_site TEXT,
    suggested_content_type TEXT,
    paa_keywords JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- SEO Content Gaps
CREATE TABLE IF NOT EXISTS seo_content_gaps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyword TEXT NOT NULL,
    volume INTEGER,
    kd INTEGER,
    funnel_stage TEXT,
    competitors_ranking JSONB DEFAULT '[]',
    top_url TEXT,
    target_site TEXT,
    competitor_source TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- SEO Content Briefs
CREATE TABLE IF NOT EXISTS seo_content_briefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyword TEXT NOT NULL,
    target_site TEXT,
    content_type TEXT,
    title TEXT,
    meta_description TEXT,
    target_word_count INTEGER,
    headings JSONB DEFAULT '[]',
    semantic_keywords JSONB DEFAULT '[]',
    faq_questions JSONB DEFAULT '[]',
    internal_links JSONB DEFAULT '[]',
    cta TEXT,
    brief_json JSONB,
    file_path TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- SEO Content Drafts
CREATE TABLE IF NOT EXISTS seo_content_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brief_id UUID REFERENCES seo_content_briefs(id),
    keyword TEXT NOT NULL,
    target_site TEXT,
    title TEXT,
    word_count INTEGER,
    file_path TEXT,
    self_critique TEXT,
    status TEXT DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- SEO Backlink Prospects
CREATE TABLE IF NOT EXISTS seo_backlink_prospects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain TEXT NOT NULL,
    page_url TEXT,
    page_title TEXT,
    page_summary TEXT,
    author_name TEXT,
    contact_email TEXT,
    dr INTEGER,
    monthly_traffic INTEGER,
    prospect_type TEXT,
    discovery_method TEXT,
    outreach_angle TEXT,
    personalisation_notes TEXT,
    links_to_competitor BOOLEAN DEFAULT false,
    competitor_names JSONB DEFAULT '[]',
    score INTEGER DEFAULT 0,
    tier TEXT,
    status TEXT DEFAULT 'new',
    created_at TIMESTAMPTZ DEFAULT now(),
    last_contacted_at TIMESTAMPTZ,
    follow_up_count INTEGER DEFAULT 0,
    reply_received BOOLEAN DEFAULT false,
    target_site TEXT
);

-- SEO Rank History
CREATE TABLE IF NOT EXISTS seo_rank_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    keyword TEXT NOT NULL,
    url TEXT,
    position REAL,
    previous_position REAL,
    impressions INTEGER,
    clicks INTEGER,
    target_site TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- SEO PR Angles
CREATE TABLE IF NOT EXISTS seo_pr_angles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    angle TEXT NOT NULL,
    target_site TEXT,
    target_publications JSONB DEFAULT '[]',
    status TEXT DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- SEO Outreach Blocklist
CREATE TABLE IF NOT EXISTS seo_outreach_blocklist (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain TEXT NOT NULL UNIQUE,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- HARO Responses
CREATE TABLE IF NOT EXISTS haro_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_topic TEXT NOT NULL,
    pitch TEXT,
    target_publication TEXT,
    status TEXT DEFAULT 'pending_review',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- LLM Cost Log
CREATE TABLE IF NOT EXISTS llm_cost_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER DEFAULT 0,
    cost_usd REAL NOT NULL,
    site TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- LLM Output Cache
CREATE TABLE IF NOT EXISTS llm_output_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key TEXT NOT NULL,
    task TEXT,
    input_key TEXT,
    result JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- SEO Outreach Emails
CREATE TABLE IF NOT EXISTS seo_outreach_emails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prospect_id UUID REFERENCES seo_backlink_prospects(id),
    subject TEXT,
    body TEXT,
    tier INTEGER,
    template_type TEXT,
    sequence_step INTEGER DEFAULT 0,
    status TEXT DEFAULT 'queued',
    sent_at TIMESTAMPTZ,
    opened BOOLEAN DEFAULT false,
    replied BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_keyword_opps_site ON seo_keyword_opportunities(target_site);
CREATE INDEX IF NOT EXISTS idx_content_gaps_site ON seo_content_gaps(target_site);
CREATE INDEX IF NOT EXISTS idx_content_briefs_site ON seo_content_briefs(target_site);
CREATE INDEX IF NOT EXISTS idx_backlink_prospects_domain ON seo_backlink_prospects(domain);
CREATE INDEX IF NOT EXISTS idx_backlink_prospects_site ON seo_backlink_prospects(target_site);
CREATE INDEX IF NOT EXISTS idx_backlink_prospects_status ON seo_backlink_prospects(status);
CREATE INDEX IF NOT EXISTS idx_rank_history_keyword_date ON seo_rank_history(keyword, date DESC);
CREATE INDEX IF NOT EXISTS idx_rank_history_site ON seo_rank_history(target_site);
CREATE INDEX IF NOT EXISTS idx_llm_cost_log_created ON llm_cost_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_cache_key ON llm_output_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_outreach_emails_prospect ON seo_outreach_emails(prospect_id);
CREATE INDEX IF NOT EXISTS idx_outreach_blocklist_domain ON seo_outreach_blocklist(domain);
