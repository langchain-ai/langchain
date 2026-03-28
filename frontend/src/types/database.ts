export interface AgentTurn {
  id: string
  session_id: string
  agent_name: string
  turn_type: string
  input: string | null
  output: string | null
  tokens_used: number
  model: string | null
  duration_ms: number | null
  created_at: string
}

export interface CronExecution {
  id: string
  job_id: string
  fired_at: string
  completed_at: string | null
  status: 'running' | 'completed' | 'failed'
  tasks_executed: number
  message_sent: boolean
  tokens_used: number
  error: string | null
  created_at: string
}

export interface LlmCostLog {
  id: string
  task_type: string
  model: string
  input_tokens: number
  output_tokens: number
  cached_tokens: number
  cost_usd: number
  site: string | null
  created_at: string
}

export interface CrmContact {
  id: string
  company_name: string
  contact_name: string | null
  contact_role: string | null
  email: string | null
  phone: string | null
  website: string | null
  city: string | null
  region: string | null
  postcode: string | null
  country: string
  category: string
  subcategory: string | null
  instagram: string | null
  facebook: string | null
  linkedin: string | null
  outreach_status: string
  outreach_segment: string | null
  score: number
  tier: string | null
  source: string | null
  tags: string[]
  notes: string | null
  created_at: string
  updated_at: string
  last_contacted_at: string | null
}

export interface CrmInteraction {
  id: string
  contact_id: string
  interaction_type: string
  direction: string
  channel: string
  subject: string | null
  body_preview: string | null
  status: string
  performed_by: string
  created_at: string
}

export interface BacklinkProspect {
  id: string
  domain: string
  page_url: string | null
  page_title: string | null
  page_summary: string | null
  author_name: string | null
  contact_email: string | null
  dr: number | null
  monthly_traffic: number | null
  prospect_type: string | null
  discovery_method: string | null
  outreach_angle: string | null
  personalisation_notes: string | null
  score: number
  tier: string | null
  status: string
  created_at: string
  last_contacted_at: string | null
  follow_up_count: number
  reply_received: boolean
  target_site: string | null
}

export interface OutreachEmail {
  id: string
  prospect_id: string
  subject: string | null
  body: string | null
  tier: number | null
  template_type: string | null
  sequence_step: number
  status: string
  sent_at: string | null
  opened: boolean
  replied: boolean
  created_at: string
}

export interface CronJob {
  id: string
  interval_hours?: number
  interval_minutes?: number
  first_delay_seconds: number
  description: string
}
