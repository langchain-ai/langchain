-- Migration 007: ralf_schedule, ralf_schedule_log, agent_memories
-- These tables are used by the agent system but had no migration file.
-- The agent's ensure_tables() creates them dynamically, but this migration
-- makes the schema explicit and idempotent.
--
-- Run via: psql $DATABASE_URL -f migrations/007_schedule_and_memory_tables.sql
-- Or paste into Supabase Dashboard → SQL Editor

-- ---------------------------------------------------------------------------
-- ralf_schedule — the recurring task schedule Ralf follows
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ralf_schedule (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cadence TEXT NOT NULL DEFAULT 'weekly',         -- daily, weekly, monthly
    day_of_week INTEGER,                            -- 0=Mon .. 6=Sun (for daily/weekly)
    day_of_month INTEGER,                           -- 1-31 (for monthly)
    skill TEXT NOT NULL,
    boost_amount INTEGER DEFAULT 30,
    label TEXT,
    description TEXT,
    site TEXT,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ralf_schedule_skill ON ralf_schedule(skill);
CREATE INDEX IF NOT EXISTS idx_ralf_schedule_active ON ralf_schedule(active);

-- ---------------------------------------------------------------------------
-- ralf_schedule_log — execution log for scheduled tasks
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ralf_schedule_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill TEXT NOT NULL,
    site TEXT,
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'done',            -- done, failed, skipped
    heartbeat_id TEXT,
    schedule_date DATE NOT NULL DEFAULT CURRENT_DATE,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ralf_schedule_log_date ON ralf_schedule_log(schedule_date DESC);
CREATE INDEX IF NOT EXISTS idx_ralf_schedule_log_skill ON ralf_schedule_log(skill);
CREATE INDEX IF NOT EXISTS idx_ralf_schedule_log_status ON ralf_schedule_log(status);

-- ---------------------------------------------------------------------------
-- agent_memories — episodic memory for learnings, corrections, preferences
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS agent_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT NOT NULL,                         -- correction, preference, decision, learning, context, activity
    content TEXT NOT NULL,
    importance INTEGER DEFAULT 5,                   -- 1-10 scale
    source TEXT DEFAULT 'telegram',                 -- telegram, worker, pulse, system
    tags JSONB DEFAULT '[]',
    recall_count INTEGER DEFAULT 0,
    last_recalled_at TIMESTAMPTZ,
    site TEXT,
    promoted BOOLEAN DEFAULT false,                 -- promoted to permanent learned lessons
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_memories_category ON agent_memories(category);
CREATE INDEX IF NOT EXISTS idx_agent_memories_created ON agent_memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memories_importance ON agent_memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memories_promoted ON agent_memories(promoted);

-- ---------------------------------------------------------------------------
-- Enable Supabase Realtime on tables the frontend subscribes to
-- ---------------------------------------------------------------------------

ALTER PUBLICATION supabase_realtime ADD TABLE agent_turns;
ALTER PUBLICATION supabase_realtime ADD TABLE cron_executions;
ALTER PUBLICATION supabase_realtime ADD TABLE llm_cost_log;
