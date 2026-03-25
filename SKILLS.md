# Skills

## Available Skills
| Skill | Description | Location |
|-------|-------------|----------|
| supabase-postgres-best-practices | Postgres performance optimisation: indexes, query patterns, RLS, connection pooling, schema design | `.agents/skills/supabase-postgres-best-practices/` |

## How to Load a Skill
1. Read `SKILL.md` in the skill directory for the overview and category listing
2. Read `AGENTS.md` for the navigation guide
3. Browse `references/` for detailed per-topic documentation
4. Reference files are loaded on-demand — read only what's relevant to the current task

## Skill Categories (supabase-postgres-best-practices)
| Priority | Category | Impact | File Prefix |
|----------|----------|--------|-------------|
| 1 | Query Performance | CRITICAL | `query-` |
| 2 | Connection Management | CRITICAL | `conn-` |
| 3 | Security & RLS | CRITICAL | `security-` |
| 4 | Schema Design | HIGH | `schema-` |
| 5 | Concurrency & Locking | MEDIUM-HIGH | `lock-` |
| 6 | Data Access Patterns | MEDIUM | `data-` |
| 7 | Monitoring & Diagnostics | LOW-MEDIUM | `monitor-` |
| 8 | Advanced Features | LOW | `advanced-` |

## When to Apply
- Writing SQL queries or designing schemas for the Supabase-backed tables
- Implementing indexes or query optimisation
- Reviewing database performance issues
- Configuring connection pooling or scaling
- Working with Row-Level Security (RLS)
