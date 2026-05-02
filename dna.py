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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from anthropic import Anthropic
from supabase import Client


MODEL = "claude-sonnet-4-6"

# Anthropic per-token pricing in USD. Source: https://www.anthropic.com/pricing
# Used by run_extraction_pass to estimate cost_usd from streamed usage data.
MODEL_PRICING_USD: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
    },
}


@dataclass(frozen=True)
class EvidenceSource:
    """Describes the kind of source text a pass extracts from.

    citation_field — the key the LLM uses inside each evidence item to
    point at the source (e.g., "video_id" for transcript-based passes,
    "comment_id" for comment-based passes). Stored as-is in the
    persisted evidence_json.

    wrapper_tag / body_tag — XML tag names for the user_message wrapper
    around each bucket item (e.g., <video><transcript>...</transcript></video>
    or <comment><text>...</text></comment>).
    """

    citation_field: str
    wrapper_tag: str
    body_tag: str


VIDEO_TRANSCRIPT = EvidenceSource("video_id", "video", "transcript")
COMMENT_TEXT = EvidenceSource("comment_id", "comment", "text")


# Bucket items are uniformly {"id": uuid, "source_text": text}. The pass-
# specific loader is responsible for translating from videos.transcript or
# comments.text into this shape.
BucketLoader = Callable[[str, Client], list[list[dict]]]


def _build_record_claims_tool(citation_field: str) -> dict:
    return {
        "name": "record_claims",
        "description": (
            "Record extracted DNA claims with citations grounded in the "
            "provided source text. Every claim must include at least one "
            "piece of evidence with the exact quoted substring from the "
            "source."
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
                                        citation_field: {"type": "string"},
                                        "exact_quote": {"type": "string"},
                                    },
                                    "required": [citation_field, "exact_quote"],
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


def _make_extract_call(citation_field: str) -> LlmCall:
    """Build an Anthropic streaming call configured for the given citation
    field. Returned closure matches the LlmCall signature.

    Streaming is required by the SDK above its "operations may take longer
    than 10 minutes" threshold. Output budget for verbose creators can run
    high; stream() unlocks the full 64K Sonnet ceiling. get_final_message()
    returns the same Message shape as messages.create().
    """
    tool = _build_record_claims_tool(citation_field)

    def _call(system_prompt: str, user_message: str) -> dict:
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
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            response = stream.get_final_message()
        # Truncation in a forced tool call leaves block.input as `{}` —
        # silent zero-claims. Surface as a hard error so the orchestrator's
        # per-bucket try/except logs it instead of treating it as success.
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Anthropic response hit max_tokens "
                f"(output={response.usage.output_tokens}); tool call was "
                "truncated. Reduce bucket size or raise max_tokens."
            )
        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and block.name == tool["name"]
            ):
                tool_input = (
                    dict(block.input) if isinstance(block.input, dict) else {}
                )
                tool_input["_anthropic_usage"] = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                return tool_input
        raise RuntimeError(
            "Anthropic response did not include record_claims tool use"
        )

    return _call


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
            tool_input = dict(block.input) if isinstance(block.input, dict) else {}
            tool_input["_anthropic_usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return tool_input
    raise RuntimeError(
        "Anthropic response did not include record_canonical_claims tool use"
    )


def _build_user_message(bucket: list[dict], evidence_source: EvidenceSource) -> str:
    parts = []
    for item in bucket:
        parts.append(
            f'<{evidence_source.wrapper_tag} id="{item["id"]}">'
            f'<{evidence_source.body_tag}>{item.get("source_text") or ""}</{evidence_source.body_tag}>'
            f'</{evidence_source.wrapper_tag}>'
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
    bucket: list[dict],
    sb: Client,
    *,
    evidence_source: EvidenceSource = VIDEO_TRANSCRIPT,
    llm_call: LlmCall | None = None,
    model_version: str | None = None,
    write_to_db: bool = True,
) -> dict[str, Any]:
    """Run extraction over a bucket of source items, verify each citation
    against the cited source's text, optionally archive-then-insert.

    bucket: list of {"id": uuid, "source_text": text|None}. id must match
    the persisted source row; source_text must equal that row's content
    so substring verification is meaningful.

    evidence_source: which kind of source the bucket holds. Drives the
    LLM tool's citation field name, the user_message XML wrapping, and
    the citation-key used for verification. Defaults to VIDEO_TRANSCRIPT
    for backwards compatibility with voice passes.

    Per-claim verification is all-or-nothing: any failing piece of
    evidence rejects the entire claim.

    write_to_db=True (default): archive prior active claims for
    (creator_id, pass_name) then insert accepted claims. Used by the
    1b.1 stub harness.

    write_to_db=False: return accepted claims to caller, no DB writes.
    Used by orchestrators that combine multiple buckets through
    synthesis before persisting.

    Returns {"written", "rejected", "rejected_reasons", "accepted_claims",
    "anthropic_usage"}. "written" == count of accepted claims regardless
    of write_to_db (the value is just whether they were also persisted).
    """
    llm = llm_call or _make_extract_call(evidence_source.citation_field)
    version = model_version or MODEL

    source_text_by_id = {item["id"]: item.get("source_text") for item in bucket}

    user_message = _build_user_message(bucket, evidence_source)
    response = llm(system_prompt, user_message)
    # Extract usage out of the LLM result so it doesn't leak into claim
    # dicts written downstream. Stubs that don't populate _anthropic_usage
    # produce None — the orchestrator's cost aggregation treats that as zero.
    usage = response.pop("_anthropic_usage", None)
    raw_claims = response.get("claims", [])

    citation_field = evidence_source.citation_field

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
            src_id = ev.get(citation_field)
            quote = ev.get("exact_quote", "")
            if src_id not in source_text_by_id:
                failure_reason = f"unknown {citation_field} {src_id}"
                break
            if not _verify_quote(quote, source_text_by_id[src_id]):
                failure_reason = f"quote not found in {citation_field} {src_id}"
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
        "anthropic_usage": usage,
    }


def synthesize_claims(
    input_claims: list[dict],
    system_prompt: str,
    *,
    evidence_source: EvidenceSource = VIDEO_TRANSCRIPT,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Merge per-bucket claims into canonical claims via source-index merging.

    The LLM emits canonical_claims with source_claim_indexes pointing into
    the flattened input list. This function then mechanically unions
    evidence from the referenced source claims (deduped by
    (citation_id, exact_quote) where citation_id is the field named in
    evidence_source.citation_field). The LLM never emits evidence, so
    hallucinated quotes are structurally impossible.

    Returns {"canonical_claims": [...], "anthropic_usage": dict | None}.
    canonical_claims is empty if input is empty or LLM produces no usable
    canonical claims. anthropic_usage is None if the LLM call was a stub
    that didn't populate _anthropic_usage.
    """
    if not input_claims:
        return {"canonical_claims": [], "anthropic_usage": None}

    llm = llm_call or _default_synthesis_call

    lines = []
    for i, c in enumerate(input_claims):
        text = (c.get("claim_text") or "").replace("</claim>", "")
        lines.append(f'<claim index="{i}">{text}</claim>')
    user_message = "\n".join(lines)

    response = llm(system_prompt, user_message)
    usage = response.pop("_anthropic_usage", None)
    canonical = response.get("canonical_claims", [])

    citation_field = evidence_source.citation_field
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
                key = (ev.get(citation_field), ev.get("exact_quote"))
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

    return {"canonical_claims": result, "anthropic_usage": usage}


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


def transcript_bucket_loader(bucket_size: int = 10) -> BucketLoader:
    """Returns a bucket loader that fetches the creator's transcripts and
    chunks them deterministically. Shared default for voice passes.

    Items have shape {"id": video_uuid, "source_text": transcript_text}.
    """

    def _load(creator_id: str, sb: Client) -> list[list[dict]]:
        pa_rows = (
            sb.table("platform_accounts")
            .select("id")
            .eq("creator_id", creator_id)
            .execute()
        ).data
        if not pa_rows:
            raise ValueError(
                f"No platform_accounts for creator_id={creator_id}"
            )
        pa_ids = [r["id"] for r in pa_rows]

        rows = (
            sb.table("videos")
            .select("id, transcript")
            .in_("platform_account_id", pa_ids)
            .not_.is_("transcript", "null")
            .execute()
        ).data
        items = [
            {"id": r["id"], "source_text": r["transcript"]}
            for r in rows
            if r.get("transcript")
        ]
        if not items:
            raise ValueError(
                f"No transcripts found for creator_id={creator_id}"
            )

        rng = random.Random(creator_id)
        rng.shuffle(items)
        return [
            items[i : i + bucket_size]
            for i in range(0, len(items), bucket_size)
        ]

    return _load


def run_extraction_pass(
    creator_id: str,
    sb: Client,
    *,
    pass_name: str,
    system_prompt: str,
    synthesis_prompt: str,
    bucket_loader: BucketLoader,
    evidence_source: EvidenceSource = VIDEO_TRANSCRIPT,
    min_bucket_success_ratio: float = 0.5,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Generic orchestrator for a bucket-and-extract DNA pass.

    Pipeline:
      1. bucket_loader fetches and shapes source items into buckets.
         Each item is {"id", "source_text"}.
      2. Per-bucket LLM extraction with citation verification — log-and-
         continue on failure. Abort if <min_bucket_success_ratio succeed.
      3. LLM synthesis via source-index merging (no hallucinated evidence).
      4. Archive prior active claims for (creator_id, pass_name).
      5. Insert canonical claims.

    evidence_source determines the LLM tool's citation field, the user-
    message XML wrapping, and the verification key. Voice passes use
    VIDEO_TRANSCRIPT; avatar uses COMMENT_TEXT.

    Returns {buckets_total, buckets_succeeded, intermediate_claims,
    canonical_claims_written, cost_usd}. cost_usd is summed across all
    real Anthropic calls; stub llm_calls produce zero.
    """
    buckets = bucket_loader(creator_id, sb)
    if not buckets:
        raise ValueError(
            f"bucket_loader returned no buckets for creator_id={creator_id}"
        )
    item_total = sum(len(b) for b in buckets)
    print(
        f"[{pass_name}] {item_total} items → {len(buckets)} buckets"
    )

    intermediate: list[dict] = []
    succeeded = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for i, bucket in enumerate(buckets):
        try:
            result = extract_claims_from_bucket(
                creator_id,
                pass_name,
                system_prompt,
                bucket,
                sb,
                evidence_source=evidence_source,
                llm_call=llm_call,
                write_to_db=False,
            )
            print(
                f"[{pass_name}] bucket {i + 1}/{len(buckets)}: "
                f"{result['written']} accepted, {result['rejected']} rejected"
            )
            intermediate.extend(result["accepted_claims"])
            usage = result.get("anthropic_usage")
            if usage:
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)
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
    synth_result = synthesize_claims(
        intermediate,
        synthesis_prompt,
        evidence_source=evidence_source,
        llm_call=llm_call,
    )
    canonical = synth_result["canonical_claims"]
    synth_usage = synth_result.get("anthropic_usage")
    if synth_usage:
        total_input_tokens += synth_usage.get("input_tokens", 0)
        total_output_tokens += synth_usage.get("output_tokens", 0)
    if not canonical:
        raise RuntimeError("Synthesis returned 0 canonical claims. Aborting.")
    print(f"[{pass_name}] synthesized → {len(canonical)} canonical claims")

    written = write_canonical_claims(
        sb, creator_id, pass_name, canonical, model_version=MODEL
    )

    pricing = MODEL_PRICING_USD.get(MODEL, {"input": 0.0, "output": 0.0})
    cost_usd = (
        total_input_tokens * pricing["input"]
        + total_output_tokens * pricing["output"]
    )

    return {
        "buckets_total": len(buckets),
        "buckets_succeeded": succeeded,
        "intermediate_claims": len(intermediate),
        "canonical_claims_written": written,
        "cost_usd": cost_usd,
    }
