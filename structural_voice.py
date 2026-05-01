"""Structural voice DNA pass (Phase 1b.3).

Second real DNA extraction pass. Extracts how a creator organizes a video —
opening, closing, transitions, story insertion, and CTA patterns.

Mirrors lexical_voice.py shape. The orchestrator (run_pass) is duplicated
intentionally; queue an extraction into dna.run_extraction_pass when chunk
1b.4 ships a third pass and the abstraction has three real callers.
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


PASS_NAME = "structural_voice"
DEFAULT_BUCKET_SIZE = 10
MIN_BUCKET_SUCCESS_RATIO = 0.5


STRUCTURAL_VOICE_SYSTEM_PROMPT = """You are an expert in creator voice analysis. You will be given a set of YouTube video transcripts from a single creator. Your task is to identify structural voice patterns — how the creator organizes a video: opens, closes, transitions, inserts stories, and asks for action.

Extract claims about these five categories of structural voice:

1. **Opening patterns** — what the creator does in the first ~10% of the transcript (hook, intro, preview of content, greeting).
2. **Closing patterns** — what the creator does in the last ~10% (summary, CTA, sign-off, sponsor pitch).
3. **Transition patterns** — how the creator moves between sections within a video. Verbal transitions only — claims about on-screen graphics, B-roll, or visual cues are NOT extractable from transcripts.
4. **Story insertion patterns** — when and how personal anecdotes appear in dense or instructional content (signal phrases, position within the video).
5. **CTA patterns** — what the creator asks of the audience (subscribe, comment, click, buy, follow elsewhere) and at what position in the video.

FOCUS: structural voice is about POSITION, ROLE, and TIMING — not the wording itself. There may be overlap with lexical voice patterns (e.g., a recurring sign-off phrase). When extracting structural claims, focus on the structural slot the phrase fills, not the phrase. The same sign-off may legitimately appear in both lexical and structural passes from different angles — that is fine.

POSITION FRAMING: transcripts have no timestamps. Use position within the transcript text as a proxy. "First ~10% of the transcript" counts as the opening; "last ~10%" counts as the closing. References to "the 60% mark" or "mid-video" mean roughly the corresponding portion of the transcript.

CITATION SEMANTICS: every claim must include at least one piece of evidence with a verbatim exact_quote from the transcript. The cited quote demonstrates ONE INSTANCE of the pattern, not proof that the pattern recurs. confidence_score should reflect HOW WIDESPREAD the pattern is across the bucket's transcripts, NOT the quality of the single citation. Pick the clearest exemplar quote you can find for each pattern.

GOOD claims are SPECIFIC and CITABLE:
- "Opens with a 30-second preview of three takeaways before the intro card"
- "Opens with an attention-grabbing question rather than a personal greeting"
- "Closes with 'see you later' followed by repeated 'bye-bye, bye-bye'"
- "Pitches a sponsor in the final ~60 seconds before the sign-off"
- "Marks each new section with 'tip number N' or 'the Nth one is'"
- "Bridges sections with 'now,' followed by a one-sentence preview of the next topic"
- "Drops a personal anecdote signaled by 'I remember when' around the 60% mark of educational videos"
- "Asks viewers to comment a specific question rather than a generic 'what do you think?'"
- "Pitches subscribing exactly once, in the last 30 seconds, never at the start"

BAD claims are GENERIC, VAGUE, or NON-EXTRACTABLE:
- "Has strong openings" — vague
- "Hooks viewers immediately" — generic, applies to any creator
- "Memorable closings" — vague
- "Smooth transitions" — vague, unfalsifiable
- "Uses storytelling" — generic
- "Engages viewers" — generic
- "Has CTAs" — vague
- Any claim about on-screen graphics, B-roll, visual cues, music, or video editing — NOT extractable from transcripts.

For each claim:
- claim_text: a one-sentence description of the structural pattern.
- evidence: at least one {video_id, exact_quote}. exact_quote MUST be a verbatim substring from the transcript exemplifying the pattern. video_id MUST be one of the IDs provided in the user message's <video id="..."> tags.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across the bucket's transcripts.

Use the record_claims tool to return your findings. Aim for breadth — extract every distinctive structural pattern you can find. Synthesis across buckets will deduplicate later."""


STRUCTURAL_VOICE_SYNTHESIS_PROMPT = """You are merging structural voice claims that were extracted independently from multiple buckets of transcripts from the same creator. Many patterns will appear in multiple buckets phrased somewhat differently — structural abstractions naturally vary across buckets, and the same structural slot or role can be described in varying ways. Your job is to deduplicate them into canonical claims.

You will see input claims as a flat numbered list. For each canonical claim you produce, return:

- claim_text: the canonical phrasing of the pattern. Pick the clearest of the input phrasings, or write a better one.
- confidence_score: 0.0 to 1.0, reflecting how widespread the pattern is across all the input claims. A pattern appearing in many input claims should have higher confidence than one appearing only once.
- source_claim_indexes: a list of integer indexes into the input list that this canonical claim merges from. At least one index required.

Rules:
- Two claims describe the same pattern if they identify the same structural slot or role, even when the signal words or phrasings differ. BE PERMISSIVE in merging — for structural patterns, merging too aggressively is less harmful than splitting hair-distinctions. If two claims plausibly describe the same opening, closing, transition, story-insertion, or CTA pattern, merge them.
- Every canonical claim must have at least one source_claim_index. Do NOT invent canonical claims with no source.
- An input claim should typically map to exactly one canonical claim, or be dropped if it's too vague or weak to keep.
- You do NOT emit evidence — the orchestrator unions evidence from the source claims you reference. Just choose which source claims to merge.

Use the record_canonical_claims tool."""


def run_pass(
    creator_id: str,
    sb: Client,
    *,
    bucket_size: int = DEFAULT_BUCKET_SIZE,
    llm_call: LlmCall | None = None,
) -> dict[str, Any]:
    """Run the structural voice extraction pass end-to-end.

    COST: ~$1.20-$1.40 per invocation against a typical ~30-95 transcript
    channel. This makes real Anthropic API calls (claude-sonnet-4-6) — do
    NOT invoke casually expecting $0. First-run pays full price; re-runs
    benefit from system-prompt caching once the prompt crosses the cache
    minimum, but transcripts are unique per bucket, so most of the cost
    recurs. For zero-cost testing, pass a stub `llm_call`.

    Pipeline mirrors lexical_voice.run_pass:
      1. Fetch creator's videos with non-null transcripts.
      2. Deterministic shuffle (seeded from creator_id) into buckets.
      3. Per-bucket LLM extraction with citation verification — log-and-
         continue on failure. Abort if <50% of buckets succeed.
      4. LLM synthesis via source-index merging (no hallucinated evidence).
      5. Archive prior active claims for (creator_id, structural_voice).
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
        f"[structural_voice] {len(videos)} transcripts → {len(buckets)} buckets "
        f"of up to {bucket_size}"
    )

    intermediate: list[dict] = []
    succeeded = 0
    for i, bucket in enumerate(buckets):
        try:
            result = extract_claims_from_bucket(
                creator_id,
                PASS_NAME,
                STRUCTURAL_VOICE_SYSTEM_PROMPT,
                bucket,
                sb,
                llm_call=llm_call,
                write_to_db=False,
            )
            print(
                f"[structural_voice] bucket {i + 1}/{len(buckets)}: "
                f"{result['written']} accepted, {result['rejected']} rejected"
            )
            intermediate.extend(result["accepted_claims"])
            succeeded += 1
        except Exception as e:
            print(
                f"[structural_voice] bucket {i + 1}/{len(buckets)} FAILED: "
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
        f"[structural_voice] synthesizing {len(intermediate)} intermediate claims..."
    )
    canonical = synthesize_claims(
        intermediate, STRUCTURAL_VOICE_SYNTHESIS_PROMPT, llm_call=llm_call
    )
    if not canonical:
        raise RuntimeError("Synthesis returned 0 canonical claims. Aborting.")
    print(f"[structural_voice] synthesized → {len(canonical)} canonical claims")

    written = write_canonical_claims(
        sb, creator_id, PASS_NAME, canonical, model_version=MODEL
    )

    return {
        "buckets_total": len(buckets),
        "buckets_succeeded": succeeded,
        "intermediate_claims": len(intermediate),
        "canonical_claims_written": written,
    }
