"""Lexical voice DNA pass (Phase 1b.2).

First real DNA extraction pass. Extracts the specific words, phrases, and
verbal habits that make a creator's voice recognizable.
"""

from __future__ import annotations

import random
from typing import Any

from supabase import Client

from dna import (
    LlmCall,
    MODEL,
    extract_claims_from_bucket,
    synthesize_claims,
    write_canonical_claims,
)


PASS_NAME = "lexical_voice"
DEFAULT_BUCKET_SIZE = 10
MIN_BUCKET_SUCCESS_RATIO = 0.5


LEXICAL_VOICE_SYSTEM_PROMPT = """You are an expert in creator voice analysis. You will be given a set of YouTube video transcripts from a single creator. Your task is to identify lexical voice patterns — the specific words, phrases, and verbal habits that make this creator's voice recognizable.

Extract claims about these four categories of lexical voice:

1. **Recurring phrases** — transitions, signature openers and closers, repeated framings (e.g., "alright so", "the way I think about it is...", "let's get into it").
2. **Distinctive vocabulary** — idioms or specific word choices that feel personal vs. generic (e.g., "absolute banger", "fundamentally", domain-specific jargon used in a personal way).
3. **Audience address** — how the creator refers to viewers ("you guys", "team", "friends", "everyone"), or whether they mostly speak without addressing the audience at all.
4. **Verbal tics** — filler patterns and sentence-starters that recur regardless of topic (e.g., "right, so...", "the thing is...", "look —", trailing "you know").

GOOD claims are SPECIFIC and CITABLE:
- "Opens videos with the phrase 'Hey friends'"
- "Uses 'fundamentally' to introduce contested or load-bearing points"
- "Refers to viewers as 'you guys' rather than 'you' or 'everyone'"

BAD claims are GENERIC, VAGUE, or NON-LEXICAL:
- "Speaks clearly" — not lexical
- "Engages viewers" — not specific, not citable
- "Uses casual language" — vague, unfalsifiable

For each claim:
- claim_text: a one-sentence description of the pattern.
- evidence: at least one {video_id, exact_quote}. exact_quote MUST be a verbatim substring from the transcript demonstrating the pattern. video_id MUST be one of the IDs provided in the user message's <video id="..."> tags.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across the transcripts in this bucket. A pattern appearing in most transcripts gets higher confidence than one appearing in a single transcript.

Use the record_claims tool to return your findings. Aim for breadth — extract every distinctive lexical pattern you can find, not just the strongest. Synthesis across buckets will deduplicate later."""


LEXICAL_VOICE_SYNTHESIS_PROMPT = """You are merging lexical voice claims that were extracted independently from multiple buckets of transcripts from the same creator. Many patterns will appear in multiple buckets phrased slightly differently. Your job is to deduplicate them into canonical claims.

You will see input claims as a flat numbered list. For each canonical claim you produce, return:

- claim_text: the canonical phrasing of the pattern. Pick the clearest of the input phrasings, or write a better one.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across all the input claims. A pattern that appears in many input claims should have higher confidence than one appearing only once.
- source_claim_indexes: a list of integer indexes into the input list that this canonical claim merges from. At least one index required.

Rules:
- Two claims describe the same pattern if they identify the same lexical phenomenon, even if phrased differently. Merge them.
- Every canonical claim must have at least one source_claim_index. Do NOT invent canonical claims with no source.
- An input claim should typically map to exactly one canonical claim, or be dropped if it's too vague to keep.
- Be conservative: prefer merging similar claims over splitting them into hair-distinctions.
- You do NOT emit evidence — the orchestrator unions evidence from the source claims you reference. Just choose which source claims to merge.

Use the record_canonical_claims tool."""


def run_pass(
    creator_id: str,
    sb: Client,
    *,
    bucket_size: int = DEFAULT_BUCKET_SIZE,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Run the lexical voice extraction pass end-to-end.

    COST: ~$1.40 per invocation against a typical ~95-transcript channel.
    This makes real Anthropic API calls (claude-sonnet-4-6) — do NOT invoke
    casually expecting $0. First-run pays full price; re-runs benefit from
    system-prompt caching but transcripts are unique per bucket, so most of
    the cost recurs. For zero-cost testing, pass a stub `llm_call`.

    Pipeline:
      1. Fetch creator's videos with non-null transcripts.
      2. Deterministic shuffle (seeded from creator_id) into buckets.
      3. Per-bucket LLM extraction with citation verification — log-and-
         continue on failure. Abort if <50% of buckets succeed.
      4. LLM synthesis via source-index merging (no hallucinated evidence).
      5. Archive prior active claims for (creator_id, lexical_voice).
      6. Insert canonical claims.

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
        f"[lexical_voice] {len(videos)} transcripts → {len(buckets)} buckets "
        f"of up to {bucket_size}"
    )

    intermediate: list[dict] = []
    succeeded = 0
    for i, bucket in enumerate(buckets):
        try:
            result = extract_claims_from_bucket(
                creator_id,
                PASS_NAME,
                LEXICAL_VOICE_SYSTEM_PROMPT,
                bucket,
                sb,
                llm_call=llm_call,
                write_to_db=False,
            )
            print(
                f"[lexical_voice] bucket {i + 1}/{len(buckets)}: "
                f"{result['written']} accepted, {result['rejected']} rejected"
            )
            intermediate.extend(result["accepted_claims"])
            succeeded += 1
        except Exception as e:
            print(
                f"[lexical_voice] bucket {i + 1}/{len(buckets)} FAILED: "
                f"{type(e).__name__}: {e}"
            )

    success_ratio = succeeded / len(buckets)
    if success_ratio < MIN_BUCKET_SUCCESS_RATIO:
        raise RuntimeError(
            f"Only {succeeded}/{len(buckets)} buckets succeeded "
            f"({success_ratio:.0%} < {MIN_BUCKET_SUCCESS_RATIO:.0%}). "
            f"Aborting before synthesis."
        )
    if not intermediate:
        raise RuntimeError("No claims extracted across all buckets. Aborting.")

    print(
        f"[lexical_voice] synthesizing {len(intermediate)} intermediate claims..."
    )
    canonical = synthesize_claims(
        intermediate, LEXICAL_VOICE_SYNTHESIS_PROMPT, llm_call=llm_call
    )
    if not canonical:
        raise RuntimeError("Synthesis returned 0 canonical claims. Aborting.")
    print(f"[lexical_voice] synthesized → {len(canonical)} canonical claims")

    written = write_canonical_claims(
        sb, creator_id, PASS_NAME, canonical, model_version=MODEL
    )

    return {
        "buckets_total": len(buckets),
        "buckets_succeeded": succeeded,
        "intermediate_claims": len(intermediate),
        "canonical_claims_written": written,
    }
