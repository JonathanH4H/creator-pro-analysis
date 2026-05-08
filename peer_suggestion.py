"""Peer suggestion engine (Phase 1c.7).

For one managed creator: pull DNA claims + existing peer pool + operator
context, ask Sonnet 4.6 to suggest 10-15 candidate YouTube channels,
validate each via YouTube forHandle (1 quota unit per call), batch-insert
to peer_suggestions sharing one generation_run_id.

Hardcoded prompt mirrors the DNA-pass + 1c.4 pattern-extraction
convention (analysis prompts in Python, content-generation prompts in
prompt_templates).
"""

from __future__ import annotations

import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from anthropic import Anthropic
from supabase import Client


MODEL = "claude-sonnet-4-6"

PRICE_INPUT_PER_TOKEN = 3.0 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 15.0 / 1_000_000
PRICE_CACHE_WRITE_PER_TOKEN = 3.75 / 1_000_000
PRICE_CACHE_READ_PER_TOKEN = 0.3 / 1_000_000

# How many DNA claims per pass to feed the prompt. Top-N by confidence.
# 10 × 4 text passes + 10 performance claims = 50 total. Keeps token cost
# bounded while giving the model a representative slice of the creator's
# voice / format / audience / performance.
CLAIMS_PER_PASS = 10
TEXT_PASS_NAMES = (
    "lexical_voice",
    "structural_voice",
    "topical_voice",
    "avatar",
)

YT_API = "https://www.googleapis.com/youtube/v3"


SYSTEM_PROMPT = """You are an expert content analyst at a creator-management agency. Your job: identify 10-15 candidate YouTube CHANNELS that this managed creator's agency should monitor as PEERS.

A peer is another channel in the managed creator's niche that:
- Posts content the audience overlaps with
- Demonstrates patterns or formats the creator could learn from or compete against
- Sits in a similar size tier (within ~10× subscribers either direction) UNLESS the operator explicitly asks for a different tier

You will see:
- The MANAGED CREATOR's identity, subscriber count, and a sample of their DNA claims (voice, format, audience, performance patterns).
- The EXISTING PEER POOL (channels already monitored — DO NOT re-suggest these under any name).
- OPERATOR CONTEXT (free-form preferences from the operator who is curating the pool). May be empty.

For EACH suggestion you produce:
- channel_handle (string): your best guess at the @handle, including the @ prefix. Use the canonical handle format YouTube would use.
- estimated_display_name (string): the channel's display name as you know it.
- estimated_subscriber_count_range (string): a rough size estimate as a range, e.g. "100K-500K", "1M-5M", "<50K". Hedge if uncertain.
- niche_fit_reasoning (string): 1-3 sentences explaining WHY this channel is in the managed creator's niche. Reference specific DNA claims or content patterns the operator should care about. Be concrete, not generic.
- audience_overlap_estimate ("high" | "medium" | "low"): your assessment of audience overlap with the managed creator. "high" = same audience and similar size or bigger. "medium" = adjacent audience, format similar. "low" = same broad space but different angle / tier.
- format_similarity (string): one short sentence describing format match (e.g., "Long-form productivity essays with personal anecdotes", "Short-form study tips with on-screen graphics").
- confidence (number 0.0-1.0): your calibrated confidence that this channel exists with this handle and that you're recommending it for the right reasons. Hedge LOWER if you're uncertain about the handle or the channel's exact niche position.

HARD REQUIREMENTS:
- Produce 10-15 suggestions. No fewer.
- Every channel_handle MUST be your genuine best guess at a real YouTube channel — do not invent handles to hit the count.
- NEVER suggest a channel that is in the existing peer pool. Do not re-suggest under a slightly different handle either.
- Diversify: include a mix of audience tiers (some larger, some similar, some smaller) and adjacent niches when relevant. Do not return 15 near-identical suggestions.
- If you are uncertain about a channel's existence, set confidence below 0.5 — but still include it if you think it's worth the operator looking up.
- If the operator context narrows the niche (e.g., "avoid hustle-culture content"), respect it. When relevant, mention how each suggestion respects the constraint.
- If the operator context conflicts with the creator's DNA-derived niche, follow the operator's explicit instructions over the inferred niche.

If the managed creator's DNA is sparse or unclear, suggest fewer and lower-confidence peers rather than padding with weak guesses. The minimum is still 10, but you can drop confidences accordingly.

Use the record_peer_suggestions tool to return your output."""


SUGGESTION_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "channel_handle": {"type": "string"},
        "estimated_display_name": {"type": "string"},
        "estimated_subscriber_count_range": {"type": "string"},
        "niche_fit_reasoning": {"type": "string"},
        "audience_overlap_estimate": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "format_similarity": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "channel_handle",
        "estimated_display_name",
        "estimated_subscriber_count_range",
        "niche_fit_reasoning",
        "audience_overlap_estimate",
        "format_similarity",
        "confidence",
    ],
}

RECORD_PEER_SUGGESTIONS_TOOL = {
    "name": "record_peer_suggestions",
    "description": (
        "Record 10-15 candidate peer YouTube channels for the managed "
        "creator with niche-fit reasoning and confidence calibration."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": SUGGESTION_OBJECT_SCHEMA,
                "minItems": 10,
                "maxItems": 15,
            },
        },
        "required": ["suggestions"],
    },
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _xml_attr(s: str | None) -> str:
    if not s:
        return ""
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _strip_handle_prefix(h: str) -> str:
    h = h.strip()
    return h[1:] if h.startswith("@") else h


def _fetch_dna_claims(sb: Client, creator_id: str) -> list[dict]:
    """Top-N claims per text pass + top-N performance claims, ordered by
    confidence_score desc. Tagged with pass for the prompt."""
    claims: list[dict] = []

    # Text passes (lexical_voice / structural_voice / topical_voice / avatar)
    for pass_name in TEXT_PASS_NAMES:
        rows = (
            sb.table("dna_claims")
            .select("pass_name, claim_text, confidence_score")
            .eq("creator_id", creator_id)
            .eq("pass_name", pass_name)
            .is_("archived_at", "null")
            .order("confidence_score", desc=True, nullsfirst=False)
            .limit(CLAIMS_PER_PASS)
            .execute()
        ).data or []
        claims.extend(rows)

    # Performance pass (separate table)
    perf_rows = (
        sb.table("dna_performance_claims")
        .select("claim_category, claim_text, confidence_score")
        .eq("creator_id", creator_id)
        .is_("archived_at", "null")
        .order("confidence_score", desc=True, nullsfirst=False)
        .limit(CLAIMS_PER_PASS)
        .execute()
    ).data or []
    for r in perf_rows:
        claims.append(
            {
                "pass_name": "performance",
                "claim_text": r.get("claim_text"),
                "confidence_score": r.get("confidence_score"),
            }
        )

    return claims


def _fetch_existing_pool(sb: Client, creator_id: str) -> list[dict]:
    """Existing peers in this creator's pool — handle, display_name, and
    aggregated niche_tags."""
    rows = (
        sb.table("creator_peer_pool")
        .select(
            "niche_tags, peer:peer_creators(handle, display_name, platform_account_id)"
        )
        .eq("creator_id", creator_id)
        .execute()
    ).data or []
    out: list[dict] = []
    for r in rows:
        peer = r.get("peer")
        # supabase-py returns either dict or list depending on shape
        if isinstance(peer, list):
            peer = peer[0] if peer else None
        if not peer:
            continue
        out.append(
            {
                "handle": peer.get("handle"),
                "display_name": peer.get("display_name"),
                "platform_account_id": peer.get("platform_account_id"),
                "niche_tags": r.get("niche_tags") or [],
            }
        )
    return out


def _fetch_managed_creator(sb: Client, creator_id: str) -> dict:
    """Creator info + first platform_account (for handle / sub count).
    Single-platform-per-creator assumption matches Phase 1a."""
    creator = (
        sb.table("creators")
        .select("id, name")
        .eq("id", creator_id)
        .single()
        .execute()
    ).data
    if not creator:
        raise ValueError(f"creator not found: {creator_id}")

    pa = (
        sb.table("platform_accounts")
        .select(
            "platform, platform_account_id, platform_handle, display_name, "
            "subscriber_count"
        )
        .eq("creator_id", creator_id)
        .limit(1)
        .execute()
    ).data
    pa_row = pa[0] if pa else {}

    return {
        "id": creator["id"],
        "name": creator["name"],
        "handle": pa_row.get("platform_handle"),
        "display_name": pa_row.get("display_name") or creator["name"],
        "platform_account_id": pa_row.get("platform_account_id"),
        "subscriber_count": pa_row.get("subscriber_count"),
    }


def _build_user_message(
    creator: dict,
    claims: list[dict],
    pool: list[dict],
    operator_context: str | None,
) -> str:
    parts: list[str] = []

    niche_tags_aggregate = sorted(
        {tag for p in pool for tag in (p.get("niche_tags") or [])}
    )

    parts.append("<managed_creator>")
    parts.append(f"  <handle>{_xml_attr(creator.get('handle') or '')}</handle>")
    parts.append(
        f"  <display_name>{_xml_attr(creator.get('display_name') or '')}</display_name>"
    )
    parts.append(
        f"  <subscriber_count>{creator.get('subscriber_count') or 0}</subscriber_count>"
    )
    parts.append(
        f"  <niche_tags_aggregate>{_xml_attr(', '.join(niche_tags_aggregate))}</niche_tags_aggregate>"
    )
    parts.append("</managed_creator>")
    parts.append("")

    parts.append("<dna_claims>")
    for c in claims:
        pass_name = c.get("pass_name") or "unknown"
        confidence = c.get("confidence_score")
        conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"
        text = c.get("claim_text") or ""
        parts.append(
            f'  <claim pass="{_xml_attr(pass_name)}" confidence="{conf_str}">{_xml_attr(text)}</claim>'
        )
    parts.append("</dna_claims>")
    parts.append("")

    parts.append("<existing_peer_pool>")
    for p in pool:
        parts.append(
            f'  <peer handle="{_xml_attr(p.get("handle") or "")}" '
            f'display_name="{_xml_attr(p.get("display_name") or "")}" />'
        )
    parts.append("</existing_peer_pool>")
    parts.append("")

    if operator_context and operator_context.strip():
        parts.append("<operator_context>")
        parts.append(operator_context.strip())
        parts.append("</operator_context>")
    else:
        parts.append("<operator_context>(none provided)</operator_context>")

    return "\n".join(parts)


def _call_anthropic(user_message: str) -> dict:
    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=8000,  # 15 suggestions × ~250 tokens each + slack
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[RECORD_PEER_SUGGESTIONS_TOOL],
        tool_choice={
            "type": "tool",
            "name": RECORD_PEER_SUGGESTIONS_TOOL["name"],
        },
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Anthropic response hit max_tokens "
            f"(output={response.usage.output_tokens}); tool call truncated."
        )

    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and block.name == RECORD_PEER_SUGGESTIONS_TOOL["name"]
        ):
            tool_input = (
                dict(block.input) if isinstance(block.input, dict) else {}
            )
            usage = response.usage
            tool_input["_anthropic"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": (
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
                "cache_read_input_tokens": (
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                ),
                "stop_reason": response.stop_reason,
                "model": MODEL,
            }
            return tool_input

    raise RuntimeError(
        f"Anthropic response did not include "
        f"{RECORD_PEER_SUGGESTIONS_TOOL['name']} tool use; "
        f"stop_reason={response.stop_reason}"
    )


def _validate_via_youtube(handle: str) -> dict | None:
    """Look up a YouTube channel by handle. Returns the channel info dict
    on success, None on failure (channel not found, quota error, etc).
    Quota cost: 1 unit per call (forHandle). NEVER falls back to
    search.list (100 units) — see PRE_LAUNCH_TODO if v1 miss rate is
    high enough to revisit.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY not set")

    handle_clean = _strip_handle_prefix(handle)
    if not handle_clean:
        return None

    try:
        resp = httpx.get(
            f"{YT_API}/channels",
            params={
                "part": "snippet,statistics",
                "forHandle": handle_clean,
                "key": api_key,
            },
            timeout=15.0,
        )
        if resp.status_code != 200:
            print(
                f"[peer_suggest] forHandle {handle_clean!r} returned "
                f"{resp.status_code}: {resp.text[:200]}",
                flush=True,
            )
            return None
        data = resp.json()
        items = data.get("items") or []
        if not items:
            return None
        item = items[0]
        snippet = item.get("snippet") or {}
        statistics = item.get("statistics") or {}
        return {
            "channel_id": item.get("id"),
            "handle": snippet.get("customUrl"),
            "display_name": snippet.get("title"),
            "subscriber_count": int(statistics.get("subscriberCount") or 0),
        }
    except Exception as e:
        print(
            f"[peer_suggest] forHandle {handle_clean!r} raised "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return None


def run_peer_suggestion(
    sb: Client, creator_id: str, operator_context: str | None
) -> dict[str, Any]:
    """One-shot peer suggestion run for a managed creator. LLM produces
    10-15 candidates; we validate each via forHandle; all rows insert as
    a single batch keyed by generation_run_id.
    """
    generation_run_id = str(uuid.uuid4())
    print(
        f"[peer_suggest] {creator_id}: starting, generation_run_id={generation_run_id}",
        flush=True,
    )

    creator = _fetch_managed_creator(sb, creator_id)
    claims = _fetch_dna_claims(sb, creator_id)
    pool = _fetch_existing_pool(sb, creator_id)
    print(
        f"[peer_suggest] {creator_id}: data assembled — "
        f"dna_claims={len(claims)}, existing_pool={len(pool)}, "
        f"operator_context={'yes' if (operator_context and operator_context.strip()) else 'no'}",
        flush=True,
    )

    try:
        user_message = _build_user_message(
            creator, claims, pool, operator_context
        )
        print(
            f"[peer_suggest] {creator_id}: user_message len={len(user_message)} chars",
            flush=True,
        )

        result = _call_anthropic(user_message)
        meta = result.pop("_anthropic")
        print(
            f"[peer_suggest] {creator_id}: anthropic call complete: "
            f"input={meta['input_tokens']}, output={meta['output_tokens']}, "
            f"cache_create={meta['cache_creation_input_tokens']}, "
            f"cache_read={meta['cache_read_input_tokens']}, "
            f"stop_reason={meta['stop_reason']}",
            flush=True,
        )

        cost = (
            meta["input_tokens"] * PRICE_INPUT_PER_TOKEN
            + meta["output_tokens"] * PRICE_OUTPUT_PER_TOKEN
            + meta["cache_creation_input_tokens"] * PRICE_CACHE_WRITE_PER_TOKEN
            + meta["cache_read_input_tokens"] * PRICE_CACHE_READ_PER_TOKEN
        )

        suggestions = result.get("suggestions") or []
        print(
            f"[peer_suggest] {creator_id}: parsed {len(suggestions)} suggestions, "
            f"validating via forHandle (1 unit each)…",
            flush=True,
        )

        rows: list[dict] = []
        validated_count = 0
        for s in suggestions:
            handle_guess = s.get("channel_handle") or ""
            validation = _validate_via_youtube(handle_guess)
            row: dict[str, Any] = {
                "creator_id": creator_id,
                "generation_run_id": generation_run_id,
                "channel_handle": handle_guess,
                "estimated_display_name": s.get("estimated_display_name"),
                "estimated_subscriber_count_range": s.get(
                    "estimated_subscriber_count_range"
                ),
                "niche_fit_reasoning": s.get("niche_fit_reasoning"),
                "audience_overlap_estimate": s.get("audience_overlap_estimate"),
                "format_similarity": s.get("format_similarity"),
                "confidence": s.get("confidence"),
                "validated": False,
                "operator_context": operator_context,
                "llm_model": meta["model"],
                "llm_cost_usd": round(cost, 4),
                "llm_input_tokens": meta["input_tokens"],
                "llm_output_tokens": meta["output_tokens"],
            }
            if validation:
                row["validated"] = True
                row["actual_channel_id"] = validation["channel_id"]
                row["actual_handle"] = validation["handle"]
                row["actual_display_name"] = validation["display_name"]
                row["actual_subscriber_count"] = validation["subscriber_count"]
                validated_count += 1
            rows.append(row)

        print(
            f"[peer_suggest] {creator_id}: validation complete — "
            f"validated={validated_count}, unvalidated={len(rows) - validated_count}",
            flush=True,
        )

        # Batch insert. Single .insert() with a list works in supabase-py.
        sb.table("peer_suggestions").insert(rows).execute()
        print(
            f"[peer_suggest] {creator_id}: inserted {len(rows)} rows "
            f"generation_run_id={generation_run_id}, ${cost:.4f} total",
            flush=True,
        )

        return {
            "ok": True,
            "generation_run_id": generation_run_id,
            "suggestions": len(rows),
            "validated": validated_count,
            "cost_usd": round(cost, 4),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(
            f"[peer_suggest] {creator_id}: FAILED: "
            f"{type(e).__name__}: {e}\n{tb}",
            flush=True,
        )
        # No row inserted on failure — TS poller will time out and the UI
        # surfaces the previous batch (or empty state). Promote to a
        # peer_suggestion_runs table if reliability becomes a concern.
        raise
