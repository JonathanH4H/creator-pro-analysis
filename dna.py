"""DNA extraction harness.

Generic primitives for any DNA pass:
- extract_claims_from_bucket: LLM extraction over a bucket of videos +
  citation verification (substring match against transcripts).
- synthesize_claims: source-index merging across bucket-level claims.
  LLM picks which input claims describe the same pattern; this module
  mechanically unions evidence, so no hallucinated quotes are possible.
- write_canonical_claims: archive-then-insert into dna_claims.

Pass-specific prompts and orchestration live in per-pass modules
(e.g., lexical_voice.py).
"""

from __future__ import annotations

import random
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
    # Streaming is required by the SDK once max_tokens crosses its
    # "operations may take longer than 10 minutes" threshold. Output budget
    # for verbose creators can run high; using stream() unlocks the full
    # 64K Sonnet ceiling. get_final_message() returns the same Message
    # shape as messages.create(), so downstream parsing is unchanged.
    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=[
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ],
        tools=[RECORD_CLAIMS_TOOL],
        tool_choice={"type": "tool", "name": "record_claims"},
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()
    # Truncation in a forced tool call leaves block.input as `{}` — silent
    # zero-claims. Surface as a hard error so the orchestrator's per-bucket
    # try/except logs it instead of treating it as a successful empty bucket.
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Anthropic response hit max_tokens (output={response.usage.output_tokens}); "
            "tool call was truncated. Reduce bucket size or raise max_tokens."
        )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_claims":
            return block.input
    raise RuntimeError("Anthropic response did not include record_claims tool use")


RECORD_CANONICAL_CLAIMS_TOOL = {
    "name": "record_canonical_claims",
    "description": (
        "Merge per-bucket claims into deduplicated canonical claims by "
        "referencing source claim indexes from the input list. Do not "
        "rewrite or invent evidence — that is handled by the orchestrator."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "canonical_claims": {
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
                        "source_claim_indexes": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                            "minItems": 1,
                        },
                    },
                    "required": ["claim_text", "source_claim_indexes"],
                },
            }
        },
        "required": ["canonical_claims"],
    },
}


def _default_synthesis_call(system_prompt: str, user_message: str) -> dict:
    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=[
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ],
        tools=[RECORD_CANONICAL_CLAIMS_TOOL],
        tool_choice={"type": "tool", "name": "record_canonical_claims"},
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Anthropic synthesis hit max_tokens (output={response.usage.output_tokens}); "
            "tool call was truncated. Reduce intermediate-claims input or raise max_tokens."
        )
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and block.name == "record_canonical_claims"
        ):
            return block.input
    raise RuntimeError(
        "Anthropic response did not include record_canonical_claims tool use"
    )


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
    write_to_db: bool = True,
) -> dict[str, Any]:
    """Run extraction over a bucket of videos, verify each citation against
    the cited transcript, optionally archive-then-insert.

    video_bucket: list of {"id": uuid, "transcript": text|None}. id must
    match videos.id; transcript must equal videos.transcript so substring
    verification is meaningful.

    Per-claim verification is all-or-nothing: any failing piece of evidence
    rejects the entire claim.

    write_to_db=True (default): archive prior active claims for
    (creator_id, pass_name) then insert accepted claims. Used by the 1b.1
    stub harness.

    write_to_db=False: return accepted claims to caller, no DB writes. Used
    by orchestrators that combine multiple buckets through synthesis before
    persisting.

    Returns {"written": int, "rejected": int, "rejected_reasons": [str],
    "accepted_claims": [dict]}. "written" == count of accepted claims
    regardless of write_to_db (the value is just whether they were also
    persisted).
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

    if accepted and write_to_db:
        write_canonical_claims(
            sb, creator_id, pass_name, accepted, model_version=version
        )

    return {
        "written": len(accepted),
        "rejected": len(rejected_reasons),
        "rejected_reasons": rejected_reasons,
        "accepted_claims": accepted,
    }


def synthesize_claims(
    input_claims: list[dict],
    system_prompt: str,
    *,
    llm_call: LlmCall | None = None,
) -> list[dict]:
    """Merge per-bucket claims into canonical claims via source-index merging.

    The LLM emits canonical_claims with source_claim_indexes pointing into
    the flattened input list. This function then mechanically unions
    evidence from the referenced source claims (deduped by
    (video_id, exact_quote)). The LLM never emits evidence, so hallucinated
    quotes are structurally impossible.

    Returns canonical claims with shape {claim_text, confidence_score,
    evidence}. Empty list if input is empty or LLM produces no usable
    canonical claims.
    """
    if not input_claims:
        return []

    llm = llm_call or _default_synthesis_call

    lines = []
    for i, c in enumerate(input_claims):
        text = (c.get("claim_text") or "").replace("</claim>", "")
        lines.append(f'<claim index="{i}">{text}</claim>')
    user_message = "\n".join(lines)

    response = llm(system_prompt, user_message)
    canonical = response.get("canonical_claims", [])

    n = len(input_claims)
    result: list[dict] = []
    for cc in canonical:
        indexes = cc.get("source_claim_indexes") or []
        valid = [i for i in indexes if isinstance(i, int) and 0 <= i < n]
        if not valid:
            continue
        evidence_seen: set[tuple] = set()
        merged_evidence: list[dict] = []
        for i in valid:
            for ev in input_claims[i].get("evidence", []):
                key = (ev.get("video_id"), ev.get("exact_quote"))
                if key in evidence_seen:
                    continue
                evidence_seen.add(key)
                merged_evidence.append(ev)
        if not merged_evidence:
            continue
        result.append(
            {
                "claim_text": cc["claim_text"],
                "confidence_score": cc.get("confidence_score"),
                "evidence": merged_evidence,
            }
        )

    return result


def write_canonical_claims(
    sb: Client,
    creator_id: str,
    pass_name: str,
    claims: list[dict],
    *,
    model_version: str | None = None,
) -> int:
    """Archive prior active claims for (creator_id, pass_name), insert the
    given claims. Returns rows inserted. Each claim must have shape
    {claim_text, evidence, confidence_score?}.
    """
    if not claims:
        return 0
    _archive_existing(sb, creator_id, pass_name)
    version = model_version or MODEL
    rows = [
        {
            "creator_id": creator_id,
            "pass_name": pass_name,
            "claim_text": c["claim_text"],
            "evidence_json": c["evidence"],
            "confidence_score": c.get("confidence_score"),
            "model_version": version,
        }
        for c in claims
    ]
    sb.table("dna_claims").insert(rows).execute()
    return len(rows)


def run_extraction_pass(
    creator_id: str,
    sb: Client,
    *,
    pass_name: str,
    system_prompt: str,
    synthesis_prompt: str,
    bucket_size: int = 10,
    min_bucket_success_ratio: float = 0.5,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Generic orchestrator for a DNA extraction pass.

    Pipeline:
      1. Fetch creator's videos with non-null transcripts.
      2. Deterministic shuffle (seeded from creator_id) into buckets.
      3. Per-bucket LLM extraction with citation verification — log-and-
         continue on failure. Abort if <min_bucket_success_ratio succeed.
      4. LLM synthesis via source-index merging (no hallucinated evidence).
      5. Archive prior active claims for (creator_id, pass_name).
      6. Insert canonical claims.

    Pass-specific modules (lexical_voice, structural_voice, topical_voice,
    ...) wrap this with their own pass_name and prompts.

    Returns counts: {buckets_total, buckets_succeeded, intermediate_claims,
    canonical_claims_written}.
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
        .select("id, transcript")
        .in_("platform_account_id", pa_ids)
        .not_.is_("transcript", "null")
        .execute()
    ).data
    videos = [v for v in videos if v.get("transcript")]
    if not videos:
        raise ValueError(f"No transcripts found for creator_id={creator_id}")

    rng = random.Random(creator_id)
    rng.shuffle(videos)
    buckets = [videos[i : i + bucket_size] for i in range(0, len(videos), bucket_size)]
    print(
        f"[{pass_name}] {len(videos)} transcripts → {len(buckets)} buckets "
        f"of up to {bucket_size}"
    )

    intermediate: list[dict] = []
    succeeded = 0
    for i, bucket in enumerate(buckets):
        try:
            result = extract_claims_from_bucket(
                creator_id,
                pass_name,
                system_prompt,
                bucket,
                sb,
                llm_call=llm_call,
                write_to_db=False,
            )
            print(
                f"[{pass_name}] bucket {i + 1}/{len(buckets)}: "
                f"{result['written']} accepted, {result['rejected']} rejected"
            )
            intermediate.extend(result["accepted_claims"])
            succeeded += 1
        except Exception as e:
            print(
                f"[{pass_name}] bucket {i + 1}/{len(buckets)} FAILED: "
                f"{type(e).__name__}: {e}"
            )

    success_ratio = succeeded / len(buckets)
    if success_ratio < min_bucket_success_ratio:
        raise RuntimeError(
            f"Only {succeeded}/{len(buckets)} buckets succeeded "
            f"({success_ratio:.0%} < {min_bucket_success_ratio:.0%}). "
            f"Aborting before synthesis."
        )
    if not intermediate:
        raise RuntimeError("No claims extracted across all buckets. Aborting.")

    print(f"[{pass_name}] synthesizing {len(intermediate)} intermediate claims...")
    canonical = synthesize_claims(
        intermediate, synthesis_prompt, llm_call=llm_call
    )
    if not canonical:
        raise RuntimeError("Synthesis returned 0 canonical claims. Aborting.")
    print(f"[{pass_name}] synthesized → {len(canonical)} canonical claims")

    written = write_canonical_claims(
        sb, creator_id, pass_name, canonical, model_version=MODEL
    )

    return {
        "buckets_total": len(buckets),
        "buckets_succeeded": succeeded,
        "intermediate_claims": len(intermediate),
        "canonical_claims_written": written,
    }
