-- Migration 006: Ensure frontend-required tables exist
-- Tables: agent_turns, cron_executions, seo_our_rankings
-- agent_turns and cron_executions were defined in 004 but may not have been run.
-- seo_our_rankings was defined in 002. This migration is idempotent.
--
-- Run via: psql $DATABASE_URL -f migrations/006_frontend_tables.sql
-- Or paste into Supabase Dashboard → SQL Editor

-- Agent turns — structured log of every agent decision/action
CREATE TABLE IF NOT EXISTS agent_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL DEFAULT 'default',
    agent_name TEXT NOT NULL DEFAULT 'ralf',
    turn_type TEXT NOT NULL,
    input TEXT,
    output TEXT,
    tokens_used INTEGER DEFAULT 0,
    model TEXT,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_turns_turn_type ON agent_turns(turn_type);
CREATE INDEX IF NOT EXISTS idx_agent_turns_created_at ON agent_turns(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_turns_session_id ON agent_turns(session_id);

-- Cron execution log — tracks every scheduled worker/pulse/heartbeat run
CREATE TABLE IF NOT EXISTS cron_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id TEXT NOT NULL,
    fired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    tasks_executed INTEGER DEFAULT 0,
    message_sent BOOLEAN DEFAULT false,
    tokens_used INTEGER DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cron_executions_job_id ON cron_executions(job_id);
CREATE INDEX IF NOT EXISTS idx_cron_executions_fired_at ON cron_executions(fired_at DESC);
CREATE INDEX IF NOT EXISTS idx_cron_executions_status ON cron_executions(status);

-- SEO rankings — our tracked keyword positions over time
CREATE TABLE IF NOT EXISTS seo_our_rankings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_site TEXT NOT NULL,
    keyword TEXT NOT NULL,
    position INTEGER,
    previous_position INTEGER,
    change INTEGER,
    url TEXT,
    volume INTEGER,
    snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_our_rankings_keyword ON seo_our_rankings(keyword, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_our_rankings_site ON seo_our_rankings(target_site, snapshot_date DESC);
