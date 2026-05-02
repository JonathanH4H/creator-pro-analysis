"""Performance DNA pass (Phase 1b.6).

Different shape from voice/avatar passes:
- Python computes a fixed set of statistics from videos table up front.
- LLM receives the stats dict + top-5 / bottom-5 video summaries as
  user message context, emits structured claims.
- Verification: each claim's statistic_value is looked up against the
  pre-computed stats dict; rejected if delta >10%. Pure dict lookup,
  no recomputation per claim.
- Writes to dna_performance_claims (separate table), not dna_claims.

Closed initial set of statistic_name values: LLM is constrained by an
enum in the tool schema. Restrictive but every claim is verifiable.
Expand the set in 1b.7+ if real claims need it.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Callable

from anthropic import Anthropic
from supabase import Client

from dna import MODEL, MODEL_PRICING_USD, LlmCall


PASS_NAME = "performance"
LONG_FORM_THRESHOLD_MINUTES = 15
TOLERANCE = 0.10
TOP_N_FOR_PROMPT = 5

CLAIM_CATEGORIES = ["top_performer", "title_pattern", "length_pattern", "topic_pattern"]


def _safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _decile_slice(sorted_values: list[float], top: bool) -> list[float]:
    """Top or bottom decile of an ascending-sorted list. At least 1 item."""
    n = max(1, len(sorted_values) // 10)
    return sorted_values[-n:] if top else sorted_values[:n]


def compute_stats(videos: list[dict]) -> dict[str, float]:
    """Compute the closed set of statistic_name values from videos."""
    views = sorted(
        [int(v["view_count"]) for v in videos if v.get("view_count") is not None]
    )
    durations_min = [
        float(v["duration_seconds"]) / 60.0
        for v in videos
        if v.get("duration_seconds") is not None
    ]
    long_form_views = [
        int(v["view_count"])
        for v in videos
        if v.get("view_count") is not None
        and v.get("duration_seconds") is not None
        and float(v["duration_seconds"]) / 60.0 > LONG_FORM_THRESHOLD_MINUTES
    ]
    short_form_views = [
        int(v["view_count"])
        for v in videos
        if v.get("view_count") is not None
        and v.get("duration_seconds") is not None
        and float(v["duration_seconds"]) / 60.0 <= LONG_FORM_THRESHOLD_MINUTES
    ]

    long_mean = _safe_mean(long_form_views)
    short_mean = _safe_mean(short_form_views)

    return {
        "mean_views": _safe_mean(views),
        "median_views": _safe_median(views),
        "top_decile_mean_views": _safe_mean(_decile_slice(views, top=True)),
        "bottom_decile_mean_views": _safe_mean(_decile_slice(views, top=False)),
        "mean_duration_minutes": _safe_mean(durations_min),
        "median_duration_minutes": _safe_median(durations_min),
        "long_form_mean_views": long_mean,
        "short_form_mean_views": short_mean,
        "long_form_short_form_view_ratio": (
            long_mean / short_mean if short_mean else 0.0
        ),
    }


STATISTIC_NAMES = list(compute_stats([]).keys())


RECORD_PERFORMANCE_CLAIMS_TOOL = {
    "name": "record_performance_claims",
    "description": (
        "Record performance DNA claims with structured numeric statistics. "
        "Each claim must cite one statistic_name from the provided enum; the "
        "statistic_value will be verified against pre-computed stats."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_category": {
                            "type": "string",
                            "enum": CLAIM_CATEGORIES,
                        },
                        "claim_text": {"type": "string"},
                        "statistic_name": {
                            "type": "string",
                            "enum": STATISTIC_NAMES,
                        },
                        "statistic_value": {"type": "number"},
                        "statistic_threshold": {"type": "number"},
                        "sample_size": {"type": "integer"},
                        "exemplar_video_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "claim_category",
                        "claim_text",
                        "statistic_name",
                        "statistic_value",
                    ],
                },
            }
        },
        "required": ["claims"],
    },
}


PERFORMANCE_SYSTEM_PROMPT = """You are an expert in creator performance analysis. You will be given pre-computed statistics for a YouTube creator's catalog plus summaries of their top-performing and bottom-performing videos. Your task is to identify performance DNA — what works, what doesn't, what patterns differentiate hits from misses.

Extract claims about these four categories:

1. **top_performer** — characteristics of the videos that vastly outperform the channel median. Identify the outliers and what makes them outliers.
2. **title_pattern** — what title structures correlate with high views (numbered lists, questions, curiosity hooks, named subjects, etc.). You can only see titles in the top/bottom summaries; reason from those.
3. **length_pattern** — does this creator perform better with short or long content? Cite the long_form_short_form_view_ratio when relevant.
4. **topic_pattern** — which topics (inferable from titles) drive views vs. flop. Generic topic claims are fine; you don't have full content access.

CRITICAL: every claim MUST cite one statistic_name from the provided list with its statistic_value. The statistic_value will be verified against the pre-computed stats — if your value differs from the recomputed value by more than 10%, the claim is rejected. Do not estimate or paraphrase the statistic; cite it exactly.

Available statistic_name values (you MUST use one of these per claim):
- mean_views, median_views
- top_decile_mean_views, bottom_decile_mean_views
- mean_duration_minutes, median_duration_minutes
- long_form_mean_views, short_form_mean_views
- long_form_short_form_view_ratio

GOOD claims:
- {category: "length_pattern", claim_text: "Long-form videos (>15 min) substantially outperform shorter content, averaging 2.1x the views.", statistic_name: "long_form_short_form_view_ratio", statistic_value: 2.14}
- {category: "top_performer", claim_text: "His top decile of videos averages 4.5x the channel median, indicating significant variance in success.", statistic_name: "top_decile_mean_views", statistic_value: 200000}

BAD claims:
- "Has good performance" — vague
- "Top videos are popular" — generic, unfalsifiable
- Citing a statistic_name not in the enum
- Citing a statistic_value that doesn't match the pre-computed stat

Optional fields per claim:
- statistic_threshold: a comparison threshold if the claim implies one (e.g., "videos above X views").
- sample_size: how many videos the claim is based on (if relevant).
- exemplar_video_ids: UUIDs of videos that exemplify the pattern (must be from the provided top/bottom summaries).
- confidence_score: 0.0 to 1.0.

Use the record_performance_claims tool. Aim for breadth across the four categories."""


def _build_user_message(stats: dict, top_videos: list[dict], bottom_videos: list[dict], video_count: int) -> str:
    stat_lines = "\n".join(f"  {k}: {v:.2f}" for k, v in stats.items())
    def _vid_lines(vids: list[dict]) -> str:
        out = []
        for v in vids:
            dur_min = (v.get("duration_seconds") or 0) / 60.0
            out.append(
                f"  - id={v['id']}: \"{(v.get('title') or '')[:120]}\" — "
                f"{v.get('view_count', 0)} views, {dur_min:.1f} min"
            )
        return "\n".join(out) if out else "  (none)"
    return (
        f"Channel video count: {video_count}\n"
        f"Long-form threshold: >{LONG_FORM_THRESHOLD_MINUTES} minutes\n\n"
        f"Pre-computed statistics:\n{stat_lines}\n\n"
        f"Top {len(top_videos)} videos by view_count:\n{_vid_lines(top_videos)}\n\n"
        f"Bottom {len(bottom_videos)} videos by view_count:\n{_vid_lines(bottom_videos)}\n\n"
        "Make claims using the record_performance_claims tool. "
        "Cite statistic_name and statistic_value exactly as listed above."
    )


def _default_performance_call(system_prompt: str, user_message: str) -> dict:
    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[RECORD_PERFORMANCE_CLAIMS_TOOL],
        tool_choice={
            "type": "tool",
            "name": RECORD_PERFORMANCE_CLAIMS_TOOL["name"],
        },
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Performance call hit max_tokens (output={response.usage.output_tokens})"
        )
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and block.name == RECORD_PERFORMANCE_CLAIMS_TOOL["name"]
        ):
            tool_input = dict(block.input) if isinstance(block.input, dict) else {}
            tool_input["_anthropic_usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return tool_input
    raise RuntimeError("Performance response did not include tool use")


def _verify_claim(claim: dict, stats: dict, valid_video_ids: set[str]) -> tuple[bool, str | None]:
    """Returns (is_valid, rejection_reason). Verifies statistic_value against
    pre-computed stats and exemplar_video_ids against the creator's videos.
    """
    stat_name = claim.get("statistic_name")
    if stat_name not in stats:
        return False, f"unknown statistic_name '{stat_name}'"
    expected = stats[stat_name]
    actual = claim.get("statistic_value")
    if actual is None:
        return False, "missing statistic_value"
    if expected == 0:
        if abs(actual) > 0.001:
            return False, f"expected ~0, got {actual}"
    else:
        rel_delta = abs(actual - expected) / abs(expected)
        if rel_delta > TOLERANCE:
            return False, (
                f"value {actual} differs from computed {expected:.2f} "
                f"by {rel_delta:.0%} (>{TOLERANCE:.0%})"
            )
    exemplars = claim.get("exemplar_video_ids") or []
    invalid = [vid for vid in exemplars if vid not in valid_video_ids]
    if invalid:
        return False, f"exemplar_video_ids not in creator's videos: {invalid}"
    return True, None


def _archive_existing(sb: Client, creator_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    sb.table("dna_performance_claims").update({"archived_at": now}).eq(
        "creator_id", creator_id
    ).is_("archived_at", "null").execute()


def run_pass(
    creator_id: str,
    sb: Client,
    *,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Run the performance extraction pass end-to-end.

    COST: ~$0.05-$0.20. Single Anthropic call (no per-bucket loop) on a
    compact stats summary. Real Anthropic API call — for zero-cost testing,
    pass a stub `llm_call`.
    """
    pa_rows = (
        sb.table("platform_accounts")
        .select("id")
        .eq("creator_id", creator_id)
        .execute()
    ).data
    if not pa_rows:
        raise ValueError(f"No platform_accounts for creator_id={creator_id}")
    pa_ids = [r["id"] for r in pa_rows]

    videos = (
        sb.table("videos")
        .select("id, title, view_count, duration_seconds")
        .in_("platform_account_id", pa_ids)
        .execute()
    ).data
    videos = [v for v in videos if v.get("view_count") is not None]
    if not videos:
        raise ValueError(
            f"No videos with view_count for creator_id={creator_id}"
        )

    stats = compute_stats(videos)
    sorted_by_views = sorted(videos, key=lambda v: v.get("view_count", 0))
    top_videos = list(reversed(sorted_by_views[-TOP_N_FOR_PROMPT:]))
    bottom_videos = sorted_by_views[:TOP_N_FOR_PROMPT]
    valid_video_ids = {v["id"] for v in videos}

    print(
        f"[performance] {len(videos)} videos. "
        f"Mean views: {stats['mean_views']:.0f}, "
        f"long/short ratio: {stats['long_form_short_form_view_ratio']:.2f}"
    )

    user_message = _build_user_message(stats, top_videos, bottom_videos, len(videos))
    llm = llm_call or _default_performance_call
    response = llm(PERFORMANCE_SYSTEM_PROMPT, user_message)
    usage = response.pop("_anthropic_usage", None)
    raw_claims = response.get("claims", [])

    accepted: list[dict] = []
    rejected_count = 0
    for claim in raw_claims:
        ok, reason = _verify_claim(claim, stats, valid_video_ids)
        if not ok:
            rejected_count += 1
            stat_name = claim.get("statistic_name")
            actual = claim.get("statistic_value")
            expected = stats.get(stat_name, "n/a")
            print(
                f"[performance] REJECTED: {claim.get('claim_category')} "
                f"\"{(claim.get('claim_text') or '')[:80]}\" — {reason} "
                f"(claimed {stat_name}={actual}, computed {expected})"
            )
            continue
        accepted.append(claim)

    print(f"[performance] {len(accepted)} accepted, {rejected_count} rejected")

    if accepted:
        _archive_existing(sb, creator_id)
        rows = [
            {
                "creator_id": creator_id,
                "claim_category": c["claim_category"],
                "claim_text": c["claim_text"],
                "statistic_name": c["statistic_name"],
                "statistic_value": c["statistic_value"],
                "statistic_threshold": c.get("statistic_threshold"),
                "sample_size": c.get("sample_size"),
                "exemplar_video_ids": c.get("exemplar_video_ids") or None,
                "confidence_score": c.get("confidence_score"),
                "model_version": MODEL,
            }
            for c in accepted
        ]
        sb.table("dna_performance_claims").insert(rows).execute()

    pricing = MODEL_PRICING_USD.get(MODEL, {"input": 0.0, "output": 0.0})
    cost_usd = 0.0
    if usage:
        cost_usd = (
            usage.get("input_tokens", 0) * pricing["input"]
            + usage.get("output_tokens", 0) * pricing["output"]
        )

    return {
        "buckets_total": 1,
        "buckets_succeeded": 1,
        "intermediate_claims": len(raw_claims),
        "canonical_claims_written": len(accepted),
        "cost_usd": cost_usd,
        "rejected_count": rejected_count,
    }
