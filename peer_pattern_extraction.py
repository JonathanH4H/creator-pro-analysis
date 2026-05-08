"""Peer pattern extraction (Phase 1c.4).

For one peer creator: fetch their full 365-day catalog (outlier + baseline
metadata, plus top-15 outlier transcripts truncated to 8000 chars), prompt
Sonnet 4.6 to identify patterns appearing DISPROPORTIONATELY in outliers
vs baseline, persist the structured output to peer_pattern_extractions.

Hardcoded prompt mirrors the DNA-pass convention (see lexical_voice.py)
rather than the generation prompt_templates table — analysis prompts live
in Python, content-generation prompts live in DB.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from anthropic import Anthropic
from supabase import Client


MODEL = "claude-sonnet-4-6"

# Sonnet 4.6 pricing per Anthropic catalog. Mirrors the constants in
# lib/anthropic/generate.ts (TS) and dna.py (Python). Duplicated rather
# than imported because dna.py is DNA-specific.
PRICE_INPUT_PER_TOKEN = 3.0 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 15.0 / 1_000_000
PRICE_CACHE_WRITE_PER_TOKEN = 3.75 / 1_000_000
PRICE_CACHE_READ_PER_TOKEN = 0.3 / 1_000_000

WINDOW_DAYS = 365
TOP_OUTLIERS_FOR_TRANSCRIPTS = 15
TRANSCRIPT_TRUNCATION_CHARS = 8000
DESCRIPTION_TRUNCATION_CHARS = 300


SYSTEM_PROMPT = """You are an expert content analyst working with a creator-management agency. You are given data about ONE YouTube peer creator — a channel the agency monitors as a benchmark for a managed creator's niche. Your job: identify what makes this peer's HIGH-PERFORMING videos different from their TYPICAL videos. Patterns that appear DISPROPORTIONATELY in outliers (high view-count) compared to baseline (typical view-count) videos.

You will see TWO sets of videos:
- BASELINE: typical videos from the last 365 days. Title, view count, duration, published date.
- OUTLIERS: videos performing at >= 2x the peer's median view count. Same metadata PLUS opening transcript content where available.

Extract four categories of patterns:

1. **title_patterns** (from titles + descriptions of all videos): repeated title structures, length conventions, hook words, punctuation, capitalization, numbers, brackets/parens. Compute frequencies across BOTH sets.

2. **hook_patterns** (from outlier transcripts only — ignore baseline since we don't have their transcripts): how outlier videos open. First sentence/paragraph framings. Narrative-first vs. statement-first. Question-first vs. claim-first. Direct address vs. third-person. Personal anecdote vs. generic claim.

3. **structural_patterns** (from outlier transcripts only): mid-video moves observable in the transcript portion provided. Recap → story → framework. Problem → diagnosis → solution. "I tried X for Y days." Numbered list traversal.

4. **topical_patterns** (from titles + descriptions + transcripts): subject-matter patterns. Specific topics, frame-of-reference, audience problem framings (e.g., "productivity for students" vs. "productivity for executives"), niche framings, controversy.

For EACH pattern you extract:
- pattern (string): one clear sentence describing the pattern.
- pattern_type ("objective" | "subjective"): "objective" if mechanically verifiable (numbers in title, ALL CAPS, length >= X chars). "subjective" if interpretive (tone, narrative shape).
- outlier_frequency (0.0-1.0): fraction of OUTLIER videos exhibiting the pattern.
- baseline_frequency (0.0-1.0): fraction of BASELINE videos exhibiting the pattern. For hook/structural patterns where baseline transcripts aren't available, set to 0.0 only if the pattern is genuinely transcript-only; if the pattern manifests in titles/descriptions you can still observe it in baseline.
- differential (number): outlier_frequency / baseline_frequency. Use 99.0 if baseline_frequency is 0 and outlier_frequency > 0.
- confidence (0.0-1.0): your calibrated confidence based on sample size + pattern clarity. Patterns observed in <5 outliers cap at 0.5. Patterns with ambiguous evidence cap at 0.6.
- evidence (string array of 2-5): short concrete examples — exact title strings or short transcript quotes — taken VERBATIM from the data. Always append the platform_video_id of the example in parentheses. Do NOT paraphrase. Do NOT cite knowledge of this creator outside the provided data.
- reasoning (string): 1-2 sentences explaining why this is a real pattern and not noise.

HARD REQUIREMENTS (drop the pattern if any fails):
- differential >= 1.5
- evidence array has at least 2 items
- every evidence item must appear VERBATIM in the data with its (platform_video_id) tag

Aim for 5-15 patterns per category. Quality > quantity. 5 strong patterns beat 15 weak ones.

Edge cases:
- If a category has fewer than 5 patterns above the differential bar: return what you have, even if 0 or 1.
- If outlier transcripts are empty (<transcript> tags missing): return EMPTY arrays for hook_patterns and structural_patterns. Title and topical can still be extracted from metadata.
- If outlier set is small (<5 videos): still extract, but cap all confidences at 0.5.

Use the record_peer_patterns tool to return your output."""


PATTERN_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "pattern_type": {
            "type": "string",
            "enum": ["objective", "subjective"],
        },
        "outlier_frequency": {"type": "number", "minimum": 0, "maximum": 1},
        "baseline_frequency": {"type": "number", "minimum": 0, "maximum": 1},
        "differential": {"type": "number", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "reasoning": {"type": "string"},
    },
    "required": [
        "pattern",
        "pattern_type",
        "outlier_frequency",
        "baseline_frequency",
        "differential",
        "confidence",
        "evidence",
        "reasoning",
    ],
}

RECORD_PEER_PATTERNS_TOOL = {
    "name": "record_peer_patterns",
    "description": (
        "Record extracted differential patterns for a peer creator across "
        "four categories: title, hook, structural, topical."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title_patterns": {
                "type": "array",
                "items": PATTERN_OBJECT_SCHEMA,
            },
            "hook_patterns": {
                "type": "array",
                "items": PATTERN_OBJECT_SCHEMA,
            },
            "structural_patterns": {
                "type": "array",
                "items": PATTERN_OBJECT_SCHEMA,
            },
            "topical_patterns": {
                "type": "array",
                "items": PATTERN_OBJECT_SCHEMA,
            },
        },
        "required": [
            "title_patterns",
            "hook_patterns",
            "structural_patterns",
            "topical_patterns",
        ],
    },
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(s: str | None, n: int) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _xml_attr(s: str | None) -> str:
    """Escape characters that would break XML attribute values. Body text
    (transcripts, descriptions) is left as-is — the model is robust to
    occasional bare angle brackets in body content; matching DNA-pass
    convention."""
    if not s:
        return ""
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _build_user_message(
    peer: dict, baseline: list[dict], outliers: list[dict]
) -> str:
    parts: list[str] = []

    handle = peer.get("handle") or peer.get("platform_account_id") or ""
    parts.append("<peer>")
    parts.append(f"  <handle>{_xml_attr(handle)}</handle>")
    parts.append(
        f"  <display_name>{_xml_attr(peer.get('display_name'))}</display_name>"
    )
    niche_tags = ", ".join(peer.get("niche_tags") or [])
    parts.append(f"  <niche_tags>{_xml_attr(niche_tags)}</niche_tags>")
    parts.append(
        f"  <subscriber_count>{peer.get('subscriber_count') or 0}</subscriber_count>"
    )
    parts.append(
        f"  <median_view_count>{peer.get('median_view_count') or 0}</median_view_count>"
    )
    parts.append(
        f"  <total_videos_in_window>{len(baseline) + len(outliers)}</total_videos_in_window>"
    )
    parts.append(f"  <outlier_count>{len(outliers)}</outlier_count>")
    parts.append("</peer>")
    parts.append("")

    parts.append("<baseline>")
    for v in baseline:
        desc = _truncate(v.get("description"), DESCRIPTION_TRUNCATION_CHARS)
        parts.append(
            f'  <video id="{_xml_attr(v["platform_video_id"])}" '
            f'title="{_xml_attr(v.get("title"))}" '
            f'view_count="{v.get("view_count") or 0}" '
            f'duration="{v.get("duration_seconds") or 0}" '
            f'published_at="{_xml_attr(v.get("published_at"))}">'
        )
        if desc:
            parts.append(f"    <description>{desc}</description>")
        parts.append("  </video>")
    parts.append("</baseline>")
    parts.append("")

    parts.append("<outliers>")
    for v in outliers:
        desc = _truncate(v.get("description"), DESCRIPTION_TRUNCATION_CHARS)
        parts.append(
            f'  <video id="{_xml_attr(v["platform_video_id"])}" '
            f'title="{_xml_attr(v.get("title"))}" '
            f'view_count="{v.get("view_count") or 0}" '
            f'outlier_score="{v.get("outlier_score") or 0}" '
            f'duration="{v.get("duration_seconds") or 0}" '
            f'published_at="{_xml_attr(v.get("published_at"))}">'
        )
        if desc:
            parts.append(f"    <description>{desc}</description>")
        transcript = v.get("transcript")
        if transcript:
            parts.append(
                f"    <transcript>{_truncate(transcript, TRANSCRIPT_TRUNCATION_CHARS)}</transcript>"
            )
        parts.append("  </video>")
    parts.append("</outliers>")

    return "\n".join(parts)


def _call_anthropic(user_message: str) -> dict:
    """Forced tool-call streaming pattern matching dna._make_extract_call.
    Streaming required for SDK long-running threshold; max_tokens guard
    catches truncated tool calls (silent zero-output failure mode).
    """
    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[RECORD_PEER_PATTERNS_TOOL],
        tool_choice={
            "type": "tool",
            "name": RECORD_PEER_PATTERNS_TOOL["name"],
        },
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Anthropic response hit max_tokens "
            f"(output={response.usage.output_tokens}); tool call truncated. "
            "Reduce input or raise max_tokens."
        )

    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and block.name == RECORD_PEER_PATTERNS_TOOL["name"]
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
        f"{RECORD_PEER_PATTERNS_TOOL['name']} tool use; "
        f"stop_reason={response.stop_reason}"
    )


def _post_validate_patterns(patterns: list[dict] | None) -> list[dict]:
    """Defense in depth — drop patterns violating the hard requirements
    in the prompt. The model should already filter these, but bad
    outputs slip through."""
    valid = []
    for p in patterns or []:
        differential = p.get("differential", 0) or 0
        evidence = p.get("evidence") or []
        if differential < 1.5:
            continue
        if len(evidence) < 2:
            continue
        valid.append(p)
    return valid


def run_peer_pattern_extraction(
    sb: Client, peer_creator_id: str
) -> dict[str, Any]:
    """One-call differential pattern extraction for a peer. Inserts a
    peer_pattern_extractions row, calls Sonnet 4.6 with the compare-style
    prompt, persists results. Raises on zero outliers (no signal to
    extract from) — caller should pre-filter.
    """
    print(f"[peer_patterns] {peer_creator_id}: starting", flush=True)

    peer_row = (
        sb.table("peer_creators")
        .select(
            "id, platform_account_id, handle, display_name, "
            "subscriber_count, median_view_count"
        )
        .eq("id", peer_creator_id)
        .single()
        .execute()
    ).data
    if not peer_row:
        raise ValueError(f"peer_creators row not found: {peer_creator_id}")

    # Niche tags come from creator_peer_pool. The peer may be in multiple
    # pools with different agency-private tag sets; we use the first pool's
    # tags as a heuristic prompt orientation. The model relies far more on
    # the actual content than on tags, so this is fine.
    pool_row = (
        sb.table("creator_peer_pool")
        .select("niche_tags")
        .eq("peer_creator_id", peer_creator_id)
        .limit(1)
        .execute()
    ).data
    niche_tags = (pool_row[0].get("niche_tags") if pool_row else None) or []
    peer_with_tags = {**peer_row, "niche_tags": niche_tags}

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    ).isoformat()

    baseline = (
        sb.table("peer_videos")
        .select(
            "platform_video_id, title, description, view_count, "
            "duration_seconds, published_at"
        )
        .eq("peer_creator_id", peer_creator_id)
        .eq("is_outlier", False)
        .gte("published_at", cutoff)
        .order("view_count", desc=True)
        .execute()
    ).data or []

    outliers_all = (
        sb.table("peer_videos")
        .select(
            "platform_video_id, title, description, view_count, "
            "outlier_score, duration_seconds, published_at, transcript"
        )
        .eq("peer_creator_id", peer_creator_id)
        .eq("is_outlier", True)
        .order("outlier_score", desc=True)
        .execute()
    ).data or []

    if len(outliers_all) == 0:
        raise ValueError(
            f"peer {peer_creator_id} has no outliers; sweep required first"
        )

    # Top-N outliers carry transcripts; the rest still appear in the prompt
    # (without transcripts) so the model knows the FULL outlier set when
    # computing frequencies.
    transcripts_used = sum(
        1
        for v in outliers_all[:TOP_OUTLIERS_FOR_TRANSCRIPTS]
        if v.get("transcript")
    )
    outliers_for_prompt = []
    for i, v in enumerate(outliers_all):
        v_copy = dict(v)
        if i >= TOP_OUTLIERS_FOR_TRANSCRIPTS:
            v_copy["transcript"] = None
        outliers_for_prompt.append(v_copy)

    print(
        f"[peer_patterns] {peer_creator_id}: data assembled — "
        f"baseline={len(baseline)}, outliers={len(outliers_all)}, "
        f"transcripts_used={transcripts_used}",
        flush=True,
    )

    inserted = (
        sb.table("peer_pattern_extractions")
        .insert(
            {
                "peer_creator_id": peer_creator_id,
                "status": "extracting",
                "extraction_model": MODEL,
                "transcripts_used": transcripts_used,
                "titles_used": len(baseline) + len(outliers_all),
                "outliers_used": len(outliers_all),
                "baseline_used": len(baseline),
            }
        )
        .execute()
    ).data
    extraction_id = inserted[0]["id"] if inserted else None
    if not extraction_id:
        raise RuntimeError("Failed to insert peer_pattern_extractions row")
    print(
        f"[peer_patterns] {peer_creator_id}: extraction_id={extraction_id}",
        flush=True,
    )

    try:
        user_message = _build_user_message(
            peer_with_tags, baseline, outliers_for_prompt
        )
        print(
            f"[peer_patterns] {peer_creator_id}: user_message len={len(user_message)} chars",
            flush=True,
        )

        result = _call_anthropic(user_message)
        meta = result.pop("_anthropic")
        print(
            f"[peer_patterns] {peer_creator_id}: anthropic call complete: "
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

        title_patterns = _post_validate_patterns(
            result.get("title_patterns")
        )
        hook_patterns = _post_validate_patterns(result.get("hook_patterns"))
        structural_patterns = _post_validate_patterns(
            result.get("structural_patterns")
        )
        topical_patterns = _post_validate_patterns(
            result.get("topical_patterns")
        )

        print(
            f"[peer_patterns] {peer_creator_id}: parsed (post-validate): "
            f"title={len(title_patterns)}, hook={len(hook_patterns)}, "
            f"structural={len(structural_patterns)}, topical={len(topical_patterns)}",
            flush=True,
        )

        sb.table("peer_pattern_extractions").update(
            {
                "status": "complete",
                "title_patterns": title_patterns,
                "hook_patterns": hook_patterns,
                "structural_patterns": structural_patterns,
                "topical_patterns": topical_patterns,
                "input_tokens": meta["input_tokens"],
                "output_tokens": meta["output_tokens"],
                "cache_creation_tokens": meta["cache_creation_input_tokens"],
                "cache_read_tokens": meta["cache_read_input_tokens"],
                "stop_reason": meta["stop_reason"],
                "cost_usd": round(cost, 4),
                "completed_at": _utcnow_iso(),
                "error": None,
            }
        ).eq("id", extraction_id).execute()

        print(
            f"[peer_patterns] {peer_creator_id}: complete: "
            f"${cost:.4f}, "
            f"{len(title_patterns)}t/{len(hook_patterns)}h/"
            f"{len(structural_patterns)}s/{len(topical_patterns)}p",
            flush=True,
        )

        return {
            "ok": True,
            "extraction_id": extraction_id,
            "cost_usd": round(cost, 4),
            "title_patterns": len(title_patterns),
            "hook_patterns": len(hook_patterns),
            "structural_patterns": len(structural_patterns),
            "topical_patterns": len(topical_patterns),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(
            f"[peer_patterns] {peer_creator_id}: FAILED: "
            f"{type(e).__name__}: {e}\n{tb}",
            flush=True,
        )
        sb.table("peer_pattern_extractions").update(
            {
                "status": "failed",
                "error": str(e)[:500],
                "completed_at": _utcnow_iso(),
            }
        ).eq("id", extraction_id).execute()
        raise
