-- Migration 007: Add RLS policies for CRM tables
-- Enables anon (frontend) read access to crm_contacts and crm_interactions.
-- Service-role key (used by backend agents) bypasses RLS automatically.
--
-- Run via: Supabase Dashboard -> SQL Editor

-- Ensure RLS is enabled (idempotent)
ALTER TABLE crm_contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE crm_interactions ENABLE ROW LEVEL SECURITY;

-- Drop policies first to make this idempotent (safe to run multiple times)
DROP POLICY IF EXISTS "anon_select_crm_contacts" ON crm_contacts;
DROP POLICY IF EXISTS "anon_select_crm_interactions" ON crm_interactions;
DROP POLICY IF EXISTS "service_role_all_crm_contacts" ON crm_contacts;
DROP POLICY IF EXISTS "service_role_all_crm_interactions" ON crm_interactions;

-- Allow anon role to SELECT all rows (single-tenant, no row-level filtering needed)
CREATE POLICY "anon_select_crm_contacts"
    ON crm_contacts
    FOR SELECT
    TO anon
    USING (true);

CREATE POLICY "anon_select_crm_interactions"
    ON crm_interactions
    FOR SELECT
    TO anon
    USING (true);

-- Defensive: explicit service_role full access in case RLS bypass is toggled off
CREATE POLICY "service_role_all_crm_contacts"
    ON crm_contacts
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "service_role_all_crm_interactions"
    ON crm_interactions
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
