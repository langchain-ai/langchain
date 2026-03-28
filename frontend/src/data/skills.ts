export type SkillCategory = 'content' | 'prospecting' | 'analytics' | 'maintenance'

export interface SkillMeta {
  name: string
  description: string
  category: SkillCategory
  priority: number
  costTier: string
  cooldownHours: number
}

export const SKILLS: SkillMeta[] = [
  {
    name: 'keyword_research',
    description: 'Discover keyword opportunities via Ahrefs for all active sites',
    category: 'content',
    priority: 90,
    costTier: 'haiku',
    cooldownHours: 24,
  },
  {
    name: 'content_gap_analysis',
    description: 'Analyse content gaps vs competitors to find unaddressed keywords',
    category: 'content',
    priority: 80,
    costTier: 'sonnet',
    cooldownHours: 48,
  },
  {
    name: 'publish_blog',
    description: 'Write and publish a blog post targeting the highest-opportunity keyword',
    category: 'content',
    priority: 75,
    costTier: 'sonnet',
    cooldownHours: 8,
  },
  {
    name: 'keyword_refresh',
    description: 'Re-run keyword research when all keywords have content',
    category: 'content',
    priority: 70,
    costTier: 'haiku',
    cooldownHours: 48,
  },
  {
    name: 'discover_prospects',
    description: 'Find new backlink prospects via Ahrefs competitor analysis',
    category: 'prospecting',
    priority: 70,
    costTier: 'haiku',
    cooldownHours: 48,
  },
  {
    name: 'score_prospects',
    description: 'Enrich and score unprocessed backlink prospects',
    category: 'prospecting',
    priority: 65,
    costTier: 'haiku',
    cooldownHours: 12,
  },
  {
    name: 'promote_to_crm',
    description: 'Promote scored prospects to CRM contacts for outreach',
    category: 'prospecting',
    priority: 60,
    costTier: 'none',
    cooldownHours: 6,
  },
  {
    name: 'track_rankings',
    description: 'Snapshot current search rankings from Ahrefs/GSC',
    category: 'analytics',
    priority: 50,
    costTier: 'none',
    cooldownHours: 24,
  },
  {
    name: 'journal_entry',
    description: 'Write a reflective journal post for ralfseo.com',
    category: 'content',
    priority: 40,
    costTier: 'sonnet',
    cooldownHours: 72,
  },
  {
    name: 'internal_linking',
    description: 'Audit and suggest internal links across blog posts',
    category: 'maintenance',
    priority: 35,
    costTier: 'haiku',
    cooldownHours: 168,
  },
  {
    name: 'memory_promotion',
    description: 'Promote high-value, frequently recalled memories into permanent learned lessons',
    category: 'maintenance',
    priority: 15,
    costTier: 'none',
    cooldownHours: 168,
  },
  {
    name: 'memory_consolidation',
    description: 'Consolidate old memories to keep the memory store efficient',
    category: 'maintenance',
    priority: 10,
    costTier: 'none',
    cooldownHours: 168,
  },
]

export type FrequencyOption = '3x_week' | '2x_week' | 'weekly' | 'fortnightly' | 'monthly' | 'off'

export const FREQUENCY_OPTIONS: { value: FrequencyOption; label: string }[] = [
  { value: '3x_week', label: '3x / week' },
  { value: '2x_week', label: '2x / week' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'fortnightly', label: 'Fortnightly' },
  { value: 'monthly', label: 'Monthly' },
  { value: 'off', label: 'Off' },
]

export const CATEGORY_ORDER: SkillCategory[] = ['content', 'prospecting', 'analytics', 'maintenance']

export const CATEGORY_LABELS: Record<SkillCategory, string> = {
  content: 'Content',
  prospecting: 'Prospecting',
  analytics: 'Analytics',
  maintenance: 'Maintenance',
}

export const CATEGORY_VARIANTS: Record<SkillCategory, string> = {
  content: 'info',
  prospecting: 'success',
  analytics: 'warning',
  maintenance: 'neutral',
}
