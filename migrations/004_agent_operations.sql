-- Migration 004: Agent operations tables (cron executions, agent turns)
-- Run via: psql $DATABASE_URL -f migrations/004_agent_operations.sql
-- Or paste into Supabase Dashboard → SQL Editor

-- Cron execution log — tracks every scheduled worker/pulse/heartbeat run
CREATE TABLE IF NOT EXISTS cron_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id TEXT NOT NULL,           -- 'worker', 'pulse', 'heartbeat'
    fired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed
    tasks_executed INTEGER DEFAULT 0,
    message_sent BOOLEAN DEFAULT false,
    tokens_used INTEGER DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cron_executions_job_id ON cron_executions(job_id);
CREATE INDEX IF NOT EXISTS idx_cron_executions_fired_at ON cron_executions(fired_at DESC);
CREATE INDEX IF NOT EXISTS idx_cron_executions_status ON cron_executions(status);

-- Agent turns — structured log of every agent decision/action
CREATE TABLE IF NOT EXISTS agent_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL DEFAULT 'default',
    agent_name TEXT NOT NULL DEFAULT 'ralf',
    turn_type TEXT NOT NULL,        -- 'user', 'cron', 'pulse', 'worker'
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
