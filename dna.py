"""DNA extraction harness (Phase 1b.1).

Provides extract_claims_from_bucket() — runs an LLM extraction pass over a
bucket of videos, verifies citations against transcripts via substring match,
and writes verified claims to dna_claims with archive-then-insert semantics.

1b.1 ships the harness only. No real pass implemented yet — verification uses
a stub llm_call. Real passes (lexical_voice, etc.) arrive in 1b.2+.

Prompt caching deliberately not added in 1b.1 since the harness only ever
invokes a stub llm_call here. Add cache_control on the system prompt + tool
schema (and possibly the user message for stable buckets) when 1b.2 ships a
real pass and caching becomes load-bearing.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable

from anthropic import Anthropic
from supabase import Client


MODEL = "claude-sonnet-4-6"


RECORD_CLAIMS_TOOL = {
    "name": "record_claims",
    "description": (
        "Record extracted DNA claims with citations grounded in the provided "
        "transcripts. Every claim must include at least one piece of evidence "
        "with the exact quoted substring from the transcript."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_text": {"type": "string"},
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "video_id": {"type": "string"},
                                    "exact_quote": {"type": "string"},
                                },
                                "required": ["video_id", "exact_quote"],
                            },
                            "minItems": 1,
                        },
                    },
                    "required": ["claim_text", "evidence"],
                },
            }
        },
        "required": ["claims"],
    },
}


LlmCall = Callable[[str, str], dict]


def _default_llm_call(system_prompt: str, user_message: str) -> dict:
    client = Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system_prompt,
        tools=[RECORD_CLAIMS_TOOL],
        tool_choice={"type": "tool", "name": "record_claims"},
        messages=[{"role": "user", "content": user_message}],
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_claims":
            return block.input
    raise RuntimeError("Anthropic response did not include record_claims tool use")


def _build_user_message(video_bucket: list[dict]) -> str:
    parts = []
    for v in video_bucket:
        parts.append(
            f'<video id="{v["id"]}"><transcript>'
            f'{v.get("transcript") or ""}'
            f'</transcript></video>'
        )
    return "\n".join(parts)


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip().lower()


def _verify_quote(quote: str, transcript: str | None) -> bool:
    # Substring existence check, case-insensitive, whitespace-collapsed.
    # Confirms the quote EXISTS in the transcript, not which occurrence the
    # LLM intended. A quote like "right, so" may match many occurrences;
    # specific quotes self-disambiguate, vague ones are a known limitation
    # to revisit when a real pass surfaces a problem.
    if not transcript:
        return False
    return _normalize(quote) in _normalize(transcript)


def _archive_existing(sb: Client, creator_id: str, pass_name: str) -> None:
    # Soft-delete pattern by design — preserves history of past extractions
    # for diff/diagnosis. Do NOT optimize by deleting archived rows.
    now = datetime.now(timezone.utc).isoformat()
    sb.table("dna_claims").update({"archived_at": now}).eq(
        "creator_id", creator_id
    ).eq("pass_name", pass_name).is_("archived_at", "null").execute()


def extract_claims_from_bucket(
    creator_id: str,
    pass_name: str,
    system_prompt: str,
    video_bucket: list[dict],
    sb: Client,
    *,
    llm_call: LlmCall | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Run extraction, verify citations, archive prior active claims for
    (creator_id, pass_name), insert verified claims.

    video_bucket: list of {"id": uuid, "transcript": text|None}. id must
    match videos.id; transcript must equal videos.transcript so substring
    verification is meaningful.

    Returns {"written": int, "rejected": int, "rejected_reasons": [str]}.
    Per-claim verification is all-or-nothing: any failing piece of evidence
    rejects the entire claim.
    """
    llm = llm_call or _default_llm_call
    version = model_version or MODEL

    transcript_by_id = {v["id"]: v.get("transcript") for v in video_bucket}

    user_message = _build_user_message(video_bucket)
    response = llm(system_prompt, user_message)
    raw_claims = response.get("claims", [])

    accepted: list[dict] = []
    rejected_reasons: list[str] = []
    for claim in raw_claims:
        evidence = claim.get("evidence") or []
        if not evidence:
            reason = "no evidence"
            print(f"[dna] rejected: {reason}: {claim.get('claim_text', '')[:80]}")
            rejected_reasons.append(reason)
            continue

        failure_reason: str | None = None
        for ev in evidence:
            vid = ev.get("video_id")
            quote = ev.get("exact_quote", "")
            if vid not in transcript_by_id:
                failure_reason = f"unknown video_id {vid}"
                break
            if not _verify_quote(quote, transcript_by_id[vid]):
                failure_reason = f"quote not found in video {vid}"
                break

        if failure_reason:
            print(f"[dna] rejected: {failure_reason}: {claim.get('claim_text', '')[:80]}")
            rejected_reasons.append(failure_reason)
            continue

        accepted.append(claim)

    if accepted:
        _archive_existing(sb, creator_id, pass_name)
        rows = [
            {
                "creator_id": creator_id,
                "pass_name": pass_name,
                "claim_text": c["claim_text"],
                "evidence_json": c["evidence"],
                "confidence_score": c.get("confidence_score"),
                "model_version": version,
            }
            for c in accepted
        ]
        sb.table("dna_claims").insert(rows).execute()

    return {
        "written": len(accepted),
        "rejected": len(rejected_reasons),
        "rejected_reasons": rejected_reasons,
    }
