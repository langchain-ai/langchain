# Context

## Project Background
Ben Shevlane runs a portfolio of home improvement websites targeting UK homeowners. The SEO agent ("Ralf") was built to automate the repetitive, time-consuming parts of SEO — keyword research, content planning, backlink outreach — so Ben can focus on product and strategy.

The project is built on a fork of the LangChain monorepo (`langchain-ai/langchain`), using LangGraph for the agent state machine and LangChain Core for base abstractions. The fork includes the custom `agents/seo_agent/` directory alongside the upstream LangChain libraries (which are not deployed).

## Domain Knowledge
- **SEO** — search engine optimisation: improving website visibility in Google search results
- **DR** — Domain Rating (Ahrefs metric, 0–100): measures a site's backlink authority
- **KD** — Keyword Difficulty (0–100): how hard it is to rank for a keyword
- **SERP** — Search Engine Results Page
- **Content gap** — a keyword competitors rank for but you don't
- **Backlink prospect** — a website that might link to your content
- **Outreach** — contacting website owners to request backlinks
- **HARO** — Help A Reporter Out: journalists requesting expert sources
- **Tier 1 vs Tier 2** — prospect quality levels; tier 1 gets personalised Opus-quality emails, tier 2 gets Sonnet-quality templates
- **Warm-up schedule** — gradually increasing outreach email volume to build sender reputation

## Stakeholders
- **Ben Shevlane** — owner, sole operator, primary Telegram user
- **Kitchen maker businesses** — listed on kitchensdirectory.co.uk, potential lead buyers
- **Homeowners** — end users of freeroomplanner.com and the kitchen cost estimator

## External References
- Supabase docs: https://supabase.com/docs
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- OpenRouter docs: https://openrouter.ai/docs
- Ahrefs API: https://ahrefs.com/api
- Google Search Console API: https://developers.google.com/webmaster-tools
- Telegram Bot API: https://core.telegram.org/bots/api
- Resend docs: https://resend.com/docs
