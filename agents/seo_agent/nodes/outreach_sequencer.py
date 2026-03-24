"""Outreach sequencer node — sends emails with safety checks and follow-up logic.

Enforces blocklist rules, send-window restrictions, daily limits, warm-up
schedules, and bounce-rate monitoring. Supports a dry-run mode (default)
that logs actions without actually sending.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import resend

from agents.seo_agent.config import (
    BLOCKED_TLD_SUFFIXES,
    MAX_BOUNCE_RATE_PERCENT,
    MAX_DAILY_OUTREACH_EMAILS,
    MIN_DAYS_BETWEEN_DOMAIN_CONTACTS,
    OUTREACH_SEND_WINDOW_END,
    OUTREACH_SEND_WINDOW_START,
    WARMUP_SCHEDULE,
)
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import llm_router, supabase_tools

logger = logging.getLogger(__name__)

# Resend sender address
RESEND_OUTREACH_FROM = os.getenv("RESEND_OUTREACH_FROM", "")

# Sequence timing: step number → days after initial send
_SEQUENCE_SCHEDULE: dict[int, int] = {
    0: 0,   # Initial email
    1: 5,   # Follow-up 1
    2: 12,  # Follow-up 2
    3: 20,  # Closing the loop
}
_EXHAUST_DAY = 21

# Keywords that signal an opt-out in replies
_OPT_OUT_KEYWORDS = frozenset({
    "unsubscribe",
    "remove me",
    "not interested",
    "stop emailing",
    "opt out",
    "take me off",
})


def _is_blocked_tld(domain: str) -> bool:
    """Check if a domain uses a blocked TLD suffix.

    Args:
        domain: The domain to check.

    Returns:
        True if the domain ends with any blocked TLD suffix.
    """
    domain_lower = domain.lower()
    return any(domain_lower.endswith(suffix) for suffix in BLOCKED_TLD_SUFFIXES)


def _is_within_send_window() -> bool:
    """Check whether the current UK time is within the allowed send window.

    Returns:
        True if the current hour (UK time) is between the configured start
        and end hours.
    """
    try:
        from zoneinfo import ZoneInfo

        uk_now = datetime.now(tz=ZoneInfo("Europe/London"))
    except ImportError:
        # Fallback: treat UTC as approximate UK time
        uk_now = datetime.now(tz=timezone.utc)

    return OUTREACH_SEND_WINDOW_START <= uk_now.hour < OUTREACH_SEND_WINDOW_END


def _get_warmup_limit() -> int:
    """Calculate the daily send limit based on the warm-up schedule.

    Determines how many weeks the outreach system has been active by
    checking the earliest sent email in Supabase.

    Returns:
        Maximum emails allowed per day for the current warm-up phase.
    """
    try:
        earliest_emails = supabase_tools.query_table(
            "seo_outreach_emails",
            filters={"status": "sent"},
            limit=1,
            order_by="sent_at",
            order_desc=False,
        )
    except Exception:
        logger.warning("Could not query earliest sent email", exc_info=True)
        return WARMUP_SCHEDULE.get("week_1", 5)

    if not earliest_emails:
        return WARMUP_SCHEDULE.get("week_1", 5)

    first_sent_str = earliest_emails[0].get("sent_at", "")
    if not first_sent_str:
        return WARMUP_SCHEDULE.get("week_1", 5)

    try:
        first_sent = datetime.fromisoformat(str(first_sent_str))
        if first_sent.tzinfo is None:
            first_sent = first_sent.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return WARMUP_SCHEDULE.get("week_1", 5)

    days_active = (datetime.now(tz=timezone.utc) - first_sent).days

    if days_active < 7:
        return WARMUP_SCHEDULE.get("week_1", 5)
    elif days_active < 14:
        return WARMUP_SCHEDULE.get("week_2", 10)
    else:
        return WARMUP_SCHEDULE.get("week_3_plus", MAX_DAILY_OUTREACH_EMAILS)


def _get_today_send_count() -> int:
    """Count how many emails have been sent today.

    Returns:
        Number of emails with status ``sent`` created today.
    """
    try:
        today_emails = supabase_tools.query_table(
            "seo_outreach_emails",
            filters={"status": "sent"},
            limit=1000,
            order_by="sent_at",
            order_desc=True,
        )
    except Exception:
        logger.warning("Could not query today's send count", exc_info=True)
        return 0

    today = datetime.now(tz=timezone.utc).date()
    count = 0
    for email in today_emails:
        sent_at = email.get("sent_at", "")
        if not sent_at:
            continue
        try:
            sent_date = datetime.fromisoformat(str(sent_at)).date()
            if sent_date == today:
                count += 1
        except (ValueError, TypeError):
            continue
    return count


def _check_bounce_rate() -> float:
    """Calculate the current hard bounce rate as a percentage.

    Returns:
        Bounce rate percentage (0.0-100.0).
    """
    try:
        all_sent = supabase_tools.query_table(
            "seo_outreach_emails",
            filters={"status": "sent"},
            limit=10000,
        )
        bounced = supabase_tools.query_table(
            "seo_outreach_emails",
            filters={"status": "bounced"},
            limit=10000,
        )
    except Exception:
        logger.warning("Could not check bounce rate", exc_info=True)
        return 0.0

    total = len(all_sent) + len(bounced)
    if total == 0:
        return 0.0

    return (len(bounced) / total) * 100.0


def _is_domain_recently_contacted(domain: str) -> bool:
    """Check if a domain was contacted within the minimum contact interval.

    Args:
        domain: The domain to check.

    Returns:
        True if the domain was contacted fewer than 90 days ago.
    """
    last_contact = supabase_tools.get_last_contact_date(domain)
    if last_contact is None:
        return False

    try:
        last_dt = datetime.fromisoformat(str(last_contact))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return False

    days_since = (datetime.now(tz=timezone.utc) - last_dt).days
    return days_since < MIN_DAYS_BETWEEN_DOMAIN_CONTACTS


def _contains_opt_out(reply_text: str) -> bool:
    """Check if a reply contains opt-out language.

    Args:
        reply_text: The reply email text to scan.

    Returns:
        True if the reply contains any opt-out keyword.
    """
    reply_lower = reply_text.lower()
    return any(keyword in reply_lower for keyword in _OPT_OUT_KEYWORDS)


def _send_email(
    to_email: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    """Send an email via the Resend API.

    Args:
        to_email: Recipient email address.
        subject: Email subject line.
        body: Email body text.

    Returns:
        Resend API response dict.

    Raises:
        Exception: If Resend API call fails.
    """
    resend.api_key = os.getenv("RESEND_API_KEY", "")

    response = resend.Emails.send({
        "from": RESEND_OUTREACH_FROM,
        "to": [to_email],
        "subject": subject,
        "text": body,
    })
    return response


def _generate_followup(
    prospect: dict[str, Any],
    sequence_step: int,
    original_subject: str,
    *,
    weekly_spend: float,
    site: str,
) -> dict[str, str]:
    """Generate a follow-up email via LLM.

    Args:
        prospect: The prospect record.
        sequence_step: Which follow-up step (1, 2, or 3).
        original_subject: The subject line of the initial email.
        weekly_spend: Current weekly LLM spend in USD.
        site: Target site name for cost logging.

    Returns:
        Dict with `subject` and `body` keys.
    """
    step_instructions = {
        1: (
            "Write follow-up 1. Add NEW value (a stat, a content idea, or a "
            "relevant update). Do NOT say 'following up on my last email' or "
            "similar. Keep it under 100 words."
        ),
        2: (
            "Write follow-up 2. Take a DIFFERENT angle entirely — mention "
            "something new you noticed about their site or content. Shorter "
            "than follow-up 1. Under 80 words."
        ),
        3: (
            "Write the 'closing the loop' email. Low pressure, friendly, "
            "brief. Say you will not follow up again unless they are interested. "
            "Under 60 words."
        ),
    }

    instruction = step_instructions.get(sequence_step, step_instructions[1])

    system = (
        "You are writing a follow-up outreach email. Rules:\n"
        "- NEVER use: collaboration, synergy, partnership, reach out\n"
        "- NEVER say 'following up on my last email'\n"
        "- Be genuinely helpful and specific\n"
        "- Subject line: keep the original thread or write a new short one\n\n"
        f"Original subject: {original_subject}\n\n"
        "Respond in this format:\n"
        "SUBJECT: <subject>\n"
        "BODY:\n<body>"
    )

    user_msg = (
        f"{instruction}\n\n"
        f"Prospect domain: {prospect.get('domain', '')}\n"
        f"Page: {prospect.get('page_url', '')}\n"
        f"Summary: {prospect.get('page_summary', '')}\n"
        f"Discovery method: {prospect.get('discovery_method', '')}"
    )

    try:
        resp = llm_router.call_llm(
            task="write_followup_email",
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            weekly_spend=weekly_spend,
            site=site,
            log_fn=supabase_tools.log_llm_cost,
        )
    except Exception:
        logger.warning("Follow-up generation failed", exc_info=True)
        return {"subject": original_subject, "body": ""}

    # Parse response
    text = resp.get("text", "")
    subject = original_subject
    body = ""
    in_body = False

    for line in text.strip().splitlines():
        if line.strip().upper().startswith("SUBJECT:"):
            subject = line.split(":", 1)[1].strip()
        elif line.strip().upper().startswith("BODY:"):
            in_body = True
        elif in_body:
            body += line + "\n"

    return {"subject": subject, "body": body.strip()}


def run_outreach_sequencer(
    state: SEOAgentState,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Execute the outreach email sequence with full safety checks.

    Enforces blocklist rules, TLD restrictions, send-window hours, daily
    limits, warm-up schedules, domain contact intervals, and bounce-rate
    thresholds before sending any email.

    When ``dry_run`` is True (the default), logs all actions without
    actually sending emails via Resend.

    Sequence timing:
        - Day 0: Initial email
        - Day 5: Follow-up 1 (new value, no "following up")
        - Day 12: Follow-up 2 (different angle)
        - Day 20: "Closing the loop" (low pressure)
        - Day 21: Mark as exhausted, enforce 90-day cooldown

    Args:
        state: The current SEO agent state.
        dry_run: If True, log actions without sending. Defaults to True.

    Returns:
        State update with `errors` and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]
    weekly_spend = state.get("llm_spend_this_week", 0.0)

    # -----------------------------------------------------------------------
    # Global safety checks
    # -----------------------------------------------------------------------

    # Check send window
    if not _is_within_send_window():
        msg = (
            f"Outside send window ({OUTREACH_SEND_WINDOW_START}:00-"
            f"{OUTREACH_SEND_WINDOW_END}:00 UK). Skipping outreach."
        )
        logger.info(msg)
        return {"errors": errors, "next_node": "END"}

    # Check bounce rate
    bounce_rate = _check_bounce_rate()
    if bounce_rate > MAX_BOUNCE_RATE_PERCENT:
        msg = (
            f"Bounce rate {bounce_rate:.1f}% exceeds maximum "
            f"{MAX_BOUNCE_RATE_PERCENT}%. All sends paused."
        )
        logger.error(msg)
        errors.append(msg)
        return {"errors": errors, "next_node": "END"}

    # Daily limit (including warm-up)
    warmup_limit = _get_warmup_limit()
    effective_limit = min(warmup_limit, MAX_DAILY_OUTREACH_EMAILS)
    today_sent = _get_today_send_count()

    if today_sent >= effective_limit:
        msg = (
            f"Daily limit reached ({today_sent}/{effective_limit}). "
            f"Skipping outreach."
        )
        logger.info(msg)
        return {"errors": errors, "next_node": "END"}

    remaining_budget = effective_limit - today_sent

    # -----------------------------------------------------------------------
    # Fetch queued emails and active sequences
    # -----------------------------------------------------------------------

    try:
        queued_emails = supabase_tools.query_table(
            "seo_outreach_emails",
            filters={"status": "queued"},
            limit=remaining_budget,
            order_by="created_at",
            order_desc=False,
        )
    except Exception:
        msg = "Failed to query queued emails from Supabase"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return {"errors": errors, "next_node": "END"}

    # Also fetch emails in active sequences that need follow-ups
    try:
        active_sequences = supabase_tools.query_table(
            "seo_outreach_emails",
            filters={"status": "sent"},
            limit=500,
            order_by="sent_at",
            order_desc=False,
        )
    except Exception:
        msg = "Failed to query active sequences from Supabase"
        logger.warning(msg, exc_info=True)
        errors.append(msg)
        active_sequences = []

    sent_count = 0

    # -----------------------------------------------------------------------
    # Process initial emails (step 0)
    # -----------------------------------------------------------------------

    for email in queued_emails:
        if sent_count >= remaining_budget:
            logger.info("Remaining daily budget exhausted")
            break

        email_id = email.get("id", "")
        prospect_id = email.get("prospect_id", "")
        to_email = ""
        domain = ""

        # Look up prospect for domain and contact info
        if prospect_id:
            try:
                prospects = supabase_tools.query_table(
                    "seo_backlink_prospects",
                    filters={"id": prospect_id},
                    limit=1,
                )
                if prospects:
                    prospect = prospects[0]
                    domain = prospect.get("domain", "")
                    to_email = prospect.get("contact_email", "")
            except Exception:
                logger.warning(
                    "Could not look up prospect %s", prospect_id, exc_info=True
                )

        if not to_email:
            logger.debug("No contact email for email %s, skipping", email_id)
            continue

        # Per-domain safety checks
        if _is_blocked_tld(domain):
            logger.info("Blocked TLD for domain %s, skipping", domain)
            try:
                supabase_tools.upsert_record(
                    "seo_outreach_emails",
                    {"id": email_id, "status": "blocked_tld"},
                )
            except Exception:
                pass
            continue

        if supabase_tools.is_domain_blocked(domain):
            logger.info("Domain %s is on blocklist, skipping", domain)
            try:
                supabase_tools.upsert_record(
                    "seo_outreach_emails",
                    {"id": email_id, "status": "blocked"},
                )
            except Exception:
                pass
            continue

        if _is_domain_recently_contacted(domain):
            logger.info(
                "Domain %s contacted within %d days, skipping",
                domain,
                MIN_DAYS_BETWEEN_DOMAIN_CONTACTS,
            )
            continue

        subject = email.get("subject", "")
        body = email.get("body", "")

        if dry_run:
            logger.info(
                "[DRY RUN] Would send to %s | Subject: %s | Domain: %s",
                to_email,
                subject,
                domain,
            )
        else:
            try:
                _send_email(to_email, subject, body)
                logger.info("Sent email to %s (domain: %s)", to_email, domain)
            except Exception:
                msg = f"Failed to send email to {to_email}"
                logger.error(msg, exc_info=True)
                errors.append(msg)
                continue

        # Update email and prospect records
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        try:
            supabase_tools.upsert_record(
                "seo_outreach_emails",
                {"id": email_id, "status": "sent", "sent_at": now_iso},
            )
            if prospect_id:
                supabase_tools.upsert_record(
                    "seo_backlink_prospects",
                    {"id": prospect_id, "last_contacted_at": now_iso},
                )
        except Exception:
            logger.warning("Failed to update send status", exc_info=True)

        sent_count += 1

    # -----------------------------------------------------------------------
    # Process follow-ups for active sequences
    # -----------------------------------------------------------------------

    now = datetime.now(tz=timezone.utc)

    for email in active_sequences:
        if sent_count >= remaining_budget:
            break

        email_id = email.get("id", "")
        prospect_id = email.get("prospect_id", "")
        current_step = email.get("sequence_step", 0) or 0
        sent_at_str = email.get("sent_at", "")

        # Skip if already replied
        if email.get("replied", False):
            continue

        if not sent_at_str:
            continue

        try:
            sent_at = datetime.fromisoformat(str(sent_at_str))
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        days_since_sent = (now - sent_at).days

        # Check for exhaustion
        if days_since_sent >= _EXHAUST_DAY and current_step >= 3:
            try:
                supabase_tools.upsert_record(
                    "seo_outreach_emails",
                    {"id": email_id, "status": "exhausted"},
                )
                if prospect_id:
                    supabase_tools.upsert_record(
                        "seo_backlink_prospects",
                        {"id": prospect_id, "status": "exhausted"},
                    )
            except Exception:
                logger.warning("Failed to mark sequence as exhausted", exc_info=True)
            continue

        # Determine next step
        next_step = current_step + 1
        if next_step not in _SEQUENCE_SCHEDULE:
            continue

        required_days = _SEQUENCE_SCHEDULE[next_step]
        if days_since_sent < required_days:
            continue

        # Reply detection — check if prospect has replied
        prospect = {}
        to_email = ""
        domain = ""

        if prospect_id:
            try:
                prospects = supabase_tools.query_table(
                    "seo_backlink_prospects",
                    filters={"id": prospect_id},
                    limit=1,
                )
                if prospects:
                    prospect = prospects[0]
                    domain = prospect.get("domain", "")
                    to_email = prospect.get("contact_email", "")

                    # Check for replies
                    if prospect.get("reply_received", False):
                        logger.info(
                            "Reply detected for %s, pausing sequence", domain
                        )
                        supabase_tools.upsert_record(
                            "seo_outreach_emails",
                            {"id": email_id, "status": "replied"},
                        )
                        continue
            except Exception:
                logger.warning(
                    "Could not look up prospect for follow-up", exc_info=True
                )
                continue

        if not to_email:
            continue

        # Per-domain safety re-check
        if _is_blocked_tld(domain) or supabase_tools.is_domain_blocked(domain):
            continue

        # Generate follow-up email
        original_subject = email.get("subject", "")
        followup = _generate_followup(
            prospect,
            next_step,
            original_subject,
            weekly_spend=weekly_spend,
            site=target_site,
        )

        if not followup["body"]:
            logger.warning("Empty follow-up generated for %s, skipping", domain)
            continue

        followup_body = followup["body"]
        # Append unsubscribe line to follow-ups as well
        followup_body += (
            "\n\nIf you'd prefer not to hear from us, just reply with "
            "'unsubscribe' and I'll remove you immediately."
        )

        if dry_run:
            logger.info(
                "[DRY RUN] Would send follow-up %d to %s | Subject: %s",
                next_step,
                to_email,
                followup["subject"],
            )
        else:
            try:
                _send_email(to_email, followup["subject"], followup_body)
                logger.info(
                    "Sent follow-up %d to %s (domain: %s)",
                    next_step,
                    to_email,
                    domain,
                )
            except Exception:
                msg = f"Failed to send follow-up {next_step} to {to_email}"
                logger.error(msg, exc_info=True)
                errors.append(msg)
                continue

        # Save follow-up email record
        now_iso = now.isoformat()
        try:
            supabase_tools.insert_record(
                "seo_outreach_emails",
                {
                    "prospect_id": prospect_id,
                    "subject": followup["subject"],
                    "body": followup_body,
                    "tier": email.get("tier"),
                    "template_type": email.get("template_type", ""),
                    "sequence_step": next_step,
                    "status": "sent",
                    "sent_at": now_iso,
                },
            )
            if prospect_id:
                supabase_tools.upsert_record(
                    "seo_backlink_prospects",
                    {
                        "id": prospect_id,
                        "last_contacted_at": now_iso,
                        "follow_up_count": next_step,
                    },
                )
        except Exception:
            logger.warning(
                "Failed to save follow-up record for %s", domain, exc_info=True
            )

        sent_count += 1

    # -----------------------------------------------------------------------
    # Reply and opt-out detection pass
    # -----------------------------------------------------------------------

    try:
        replied_prospects = supabase_tools.query_table(
            "seo_backlink_prospects",
            filters={"reply_received": True},
            limit=100,
        )
    except Exception:
        logger.warning("Could not check for replied prospects", exc_info=True)
        replied_prospects = []

    for prospect in replied_prospects:
        domain = prospect.get("domain", "")
        personalisation_notes = prospect.get("personalisation_notes", "") or ""

        # Check if the reply contains opt-out language
        if _contains_opt_out(personalisation_notes):
            logger.info("Opt-out detected for domain %s, adding to blocklist", domain)
            try:
                supabase_tools.insert_record(
                    "seo_outreach_blocklist",
                    {
                        "domain": domain,
                        "reason": "Recipient opted out via reply",
                    },
                )
            except Exception:
                msg = f"Failed to add {domain} to blocklist after opt-out"
                logger.warning(msg, exc_info=True)
                errors.append(msg)

    mode_label = "DRY RUN" if dry_run else "LIVE"
    logger.info(
        "[%s] Outreach sequencer complete: %d emails processed",
        mode_label,
        sent_count,
    )

    return {
        "errors": errors,
        "next_node": "END",
    }
