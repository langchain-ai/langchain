-- Outreach system tables for Instantly V2 campaign management
-- Tracks targets, email campaigns, replies, and confirmed backlinks

CREATE TABLE IF NOT EXISTS outreach_targets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id TEXT NOT NULL,
    site_name TEXT NOT NULL,
    url TEXT NOT NULL,
    contact_email TEXT,
    contact_name TEXT,
    article_url TEXT,
    article_title TEXT,
    outreach_type TEXT,
    domain_rating INTEGER,
    notes TEXT,
    status TEXT DEFAULT 'queued',
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(url, site_id)
);

CREATE TABLE IF NOT EXISTS outreach_emails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id TEXT NOT NULL,
    target_id UUID REFERENCES outreach_targets(id),
    instantly_campaign_id TEXT,
    instantly_campaign_name TEXT,
    outreach_type TEXT,
    emails_added INTEGER,
    duplicates_skipped INTEGER,
    launched BOOLEAN DEFAULT false,
    daily_limit INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS outreach_replies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id TEXT,
    target_id UUID REFERENCES outreach_targets(id),
    instantly_campaign_id TEXT,
    from_address TEXT,
    subject TEXT,
    body TEXT,
    sentiment TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS outreach_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id TEXT NOT NULL,
    target_id UUID REFERENCES outreach_targets(id),
    target_url TEXT NOT NULL,
    link_url TEXT NOT NULL,
    anchor_text TEXT,
    domain_rating INTEGER,
    do_follow BOOLEAN DEFAULT true,
    confirmed_at TIMESTAMPTZ DEFAULT now(),
    last_checked_at TIMESTAMPTZ DEFAULT now(),
    is_live BOOLEAN DEFAULT true
);

CREATE INDEX IF NOT EXISTS idx_outreach_targets_site ON outreach_targets(site_id);
CREATE INDEX IF NOT EXISTS idx_outreach_targets_status ON outreach_targets(status);
CREATE INDEX IF NOT EXISTS idx_outreach_emails_site ON outreach_emails(site_id);
CREATE INDEX IF NOT EXISTS idx_outreach_replies_campaign ON outreach_replies(instantly_campaign_id);
CREATE INDEX IF NOT EXISTS idx_outreach_links_site ON outreach_links(site_id);
